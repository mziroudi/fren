// route_trace: log which experts the MoE router selects, per token and per layer.
//
// Uses llama.cpp's cb_eval callback to read the "ffn_moe_topk-<layer>" node built in
// llm_graph_context::build_moe_ffn (src/llama-graph.cpp). That node is the
// [n_expert_used, n_tokens] I32 view of the router's argsort.
//
// For each input it applies the model's chat template, runs the prompt as one batch
// (phase "prompt"), then greedily generates up to -n tokens one at a time (phase "gen").
//
// Output (TSV, one row per token per MoE layer):
//   task  input_id  token_pos  phase  layer  experts(comma separated, in router order)
// Plus a JSONL file with the generated text for each input.
//
// usage: route_trace -m model.gguf -i inputs.jsonl -o trace.tsv -g gens.jsonl [-n 200] [-t threads]
//   inputs.jsonl rows need: task, input_id, prompt

#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

struct trace_state {
    FILE *      out = nullptr;
    std::string task;
    int         input_id = 0;
    const char * phase = "prompt";
    long        batch_pos0 = 0;   // position of the first token in the current batch
    int         batch_n    = 0;   // number of tokens in the current batch
    long        rows = 0;
    int         k = -1;
};

static bool trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    static const char prefix[] = "ffn_moe_topk-";
    if (strncmp(t->name, prefix, sizeof(prefix) - 1) != 0) {
        return ask ? false : true;
    }
    if (ask) {
        return true; // we want this node's data
    }
    auto * st = (trace_state *) user_data;
    const int layer    = atoi(t->name + sizeof(prefix) - 1);
    const int k        = (int) t->ne[0];
    const int n_tokens = (int) t->ne[1];
    if (t->type != GGML_TYPE_I32) {
        fprintf(stderr, "unexpected type for %s\n", t->name);
        exit(1);
    }
    if (st->k < 0) st->k = k;

    std::vector<char> buf;
    const char * base;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        base = (const char *) t->data;
    } else {
        // copy the parent region the view spans
        const size_t nbytes = t->nb[1] * (n_tokens - 1) + t->nb[0] * k;
        buf.resize(nbytes);
        ggml_backend_tensor_get(t, buf.data(), 0, nbytes);
        base = buf.data();
    }

    // llama.cpp computes the last layer only for tokens whose output is requested (the
    // final token of the batch), so a node can hold fewer tokens than the batch: they are
    // always the last n_tokens of the batch.
    const long pos = st->batch_pos0 + (st->batch_n - n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        const int32_t * row = (const int32_t *) (base + i * t->nb[1]);
        fprintf(st->out, "%s\t%d\t%ld\t%s\t%d\t", st->task.c_str(), st->input_id, pos + i, st->phase, layer);
        for (int j = 0; j < k; ++j) {
            fprintf(st->out, j ? ",%d" : "%d", row[j]);
        }
        fputc('\n', st->out);
        st->rows++;
    }
    return true;
}

static std::string token_str(const llama_vocab * vocab, llama_token id) {
    char buf[256];
    int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    return n > 0 ? std::string(buf, n) : std::string();
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    std::string model_path, in_path, out_path, gen_path;
    int n_gen = 200, n_threads = 4, limit = -1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "-m") model_path = next();
        else if (a == "-i") in_path = next();
        else if (a == "-o") out_path = next();
        else if (a == "-g") gen_path = next();
        else if (a == "-n") n_gen = std::stoi(next());
        else if (a == "-t") n_threads = std::stoi(next());
        else if (a == "--limit") limit = std::stoi(next());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || in_path.empty() || out_path.empty() || gen_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -i inputs.jsonl -o trace.tsv -g gens.jsonl [-n 200] [-t 4]\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only: routing does not depend on the device
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) { fprintf(stderr, "model has no chat template\n"); return 1; }

    trace_state st;
    st.out = fopen(out_path.c_str(), "w");
    if (!st.out) { fprintf(stderr, "cannot open %s\n", out_path.c_str()); return 1; }
    fprintf(st.out, "task\tinput_id\ttoken_pos\tphase\tlayer\texperts\n");
    std::ofstream gens(gen_path);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 4096;
    cparams.n_batch         = 4096;
    cparams.n_ubatch        = 4096;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.cb_eval           = trace_cb;
    cparams.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    llama_batch_ext * batch = llama_batch_ext_init(ctx);

    std::ifstream in(in_path);
    std::string line;
    int n_done = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (limit >= 0 && n_done >= limit) break;
        json row = json::parse(line);
        st.task     = row["task"].get<std::string>();
        st.input_id = row["input_id"].get<int>();
        const std::string user = row["prompt"].get<std::string>();

        llama_chat_message msg = { "user", user.c_str() };
        int len = llama_chat_apply_template(tmpl, &msg, 1, true, nullptr, 0);
        std::string formatted(len, '\0');
        llama_chat_apply_template(tmpl, &msg, 1, true, formatted.data(), len);

        int n_prompt = -llama_tokenize(vocab, formatted.c_str(), formatted.size(), nullptr, 0, true, true);
        std::vector<llama_token> toks(n_prompt);
        llama_tokenize(vocab, formatted.c_str(), formatted.size(), toks.data(), n_prompt, true, true);
        if (n_prompt + n_gen > (int) cparams.n_ctx) { fprintf(stderr, "input too long\n"); return 1; }

        llama_memory_clear(llama_get_memory(ctx), true);
        llama_sampler_reset(smpl);

        const int64_t t0 = ggml_time_us();
        st.phase = "prompt";
        st.batch_pos0 = 0;
        st.batch_n = n_prompt;
        llama_batch_ext_clear(batch);
        for (int i = 0; i < n_prompt; ++i) {
            const int32_t idx = llama_batch_ext_add_token(batch, 0, toks[i]);
            const llama_pos p = i;
            llama_batch_ext_set_pos(batch, idx, &p);
        }
        llama_batch_ext_set_output_logits(batch, n_prompt - 1, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) { fprintf(stderr, "decode failed\n"); return 1; }
        const int64_t t1 = ggml_time_us();

        st.phase = "gen";
        std::string out_text;
        int n_pos = n_prompt, n_out = 0;
        bool hit_eog = false;
        while (n_out < n_gen) {
            llama_token id = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, id)) { hit_eog = true; break; }
            out_text += token_str(vocab, id);
            // the routing of a generated token is recorded when that token is fed back in
            llama_batch_ext_clear(batch);
            const int32_t idx = llama_batch_ext_add_token(batch, 0, id);
            const llama_pos p = n_pos;
            llama_batch_ext_set_pos(batch, idx, &p);
            llama_batch_ext_set_output_logits(batch, 0, true);
            st.batch_pos0 = n_pos;
            st.batch_n = 1;
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) { fprintf(stderr, "decode failed\n"); return 1; }
            n_pos++;
            n_out++;
        }
        const int64_t t2 = ggml_time_us();

        json g = {
            {"task", st.task}, {"input_id", st.input_id},
            {"n_prompt_tokens", n_prompt}, {"n_gen_tokens", n_out}, {"hit_eog", hit_eog},
            {"prompt_s", (t1 - t0) / 1e6}, {"gen_s", (t2 - t1) / 1e6},
            {"output", out_text},
        };
        gens << g.dump() << "\n";
        gens.flush();
        fflush(st.out);
        n_done++;
        fprintf(stderr, "[%s %d] prompt %d tok %.1fs, gen %d tok %.1fs (%.1f tok/s), rows %ld\n",
                st.task.c_str(), st.input_id, n_prompt, (t1 - t0) / 1e6, n_out, (t2 - t1) / 1e6,
                n_out / ((t2 - t1) / 1e6 + 1e-9), st.rows);
    }

    fclose(st.out);
    llama_batch_ext_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
