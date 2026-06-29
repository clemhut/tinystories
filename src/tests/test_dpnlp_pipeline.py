import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest import mock


class PipelineSpecTests(unittest.TestCase):
    def test_model_registry_exposes_four_readme_sizes(self) -> None:
        from src.dpnlp_pipeline.config import MODEL_SPECS

        self.assertTrue({"small", "medium", "large", "xl"}.issubset(MODEL_SPECS.keys()))
        self.assertEqual(MODEL_SPECS["small"].target_params, 3_000_000)
        self.assertEqual(MODEL_SPECS["medium"].target_params, 10_000_000)
        self.assertEqual(MODEL_SPECS["large"].target_params, 25_000_000)
        self.assertEqual(MODEL_SPECS["xl"].target_params, 50_000_000)

    def test_pipeline_defaults_match_paper_guidance(self) -> None:
        from src.dpnlp_pipeline.config import PipelineDefaults

        defaults = PipelineDefaults()
        self.assertEqual(defaults.context_length, 512)
        self.assertEqual(defaults.attention_window, 256)
        self.assertEqual(defaults.vocab_size, 10_000)
        self.assertEqual(defaults.gpt_eval_samples_per_prompt, 10)
        self.assertEqual(defaults.generation_temperature, 0.0)
        self.assertEqual(defaults.gpt_eval_temperature, 1.0)

    def test_dataset_paths_point_to_repo_tinystories_files(self) -> None:
        from src.dpnlp_pipeline.config import PipelinePaths

        paths = PipelinePaths.from_repo_root(Path.cwd())
        self.assertEqual(paths.train_path, Path.cwd() / "datasets" / "tinystories" / "TinyStories-train.txt")
        self.assertEqual(paths.valid_path, Path.cwd() / "datasets" / "tinystories" / "TinyStories-valid.txt")
        self.assertEqual(paths.vocab_path, Path.cwd() / "artifacts" / "dpnlp_pipeline" / "top_10k_vocab.json")

    def test_model_specs_are_compatible_with_rope_and_diff_attention(self) -> None:
        from src.dpnlp_pipeline.config import MODEL_SPECS

        for name, spec in MODEL_SPECS.items():
            with self.subTest(model=name):
                self.assertEqual(spec.d_model % (2 * spec.n_heads), 0)
                d_head = spec.d_model // spec.n_heads // 2
                self.assertEqual(d_head % 2, 0)
                self.assertEqual(spec.n_heads % (spec.n_heads // 2), 0)

    def test_model_specs_land_near_target_parameter_counts(self) -> None:
        from src.dpnlp_pipeline.config import MODEL_SPECS, PipelineDefaults
        from src.dpnlp_pipeline.modeling import build_model

        defaults = PipelineDefaults()
        tolerances = {
            "xs": 400_000,
            "small": 50_000,
            "medium": 300_000,
            "large": 400_000,
            "xl": 400_000,
        }

        for name, spec in MODEL_SPECS.items():
            with self.subTest(model=name):
                model = build_model(defaults.vocab_size + 3, spec, defaults.context_length)
                count = model.get_param_number()
                self.assertLessEqual(abs(count - spec.target_params), tolerances[name])


class RestrictedTokenizerTests(unittest.TestCase):
    def test_restricts_vocab_and_keeps_special_tokens(self) -> None:
        from src.dpnlp_pipeline.tokenizer import RestrictedTokenizer

        class FakeBackendTokenizer:
            vocab_size = 6
            eos_token = "<eos>"
            bos_token = "<bos>"
            pad_token = "<pad>"
            eos_token_id = 4
            bos_token_id = 3
            pad_token_id = 5
            special_tokens_map = {"eos_token": "<eos>", "bos_token": "<bos>", "pad_token": "<pad>"}

            def __call__(self, text, truncation=False, return_attention_mask=False, return_tensors=None):
                ids = [0, 1, 2, 3, 4, 5]
                if return_tensors == "pt":
                    import torch

                    return {"input_ids": torch.tensor([ids])}
                return {"input_ids": ids}

            def decode(self, input_ids):
                return "|".join(str(value) for value in input_ids)

        tokenizer = RestrictedTokenizer(FakeBackendTokenizer(), kept_token_ids=[2, 0, 1])
        ids = tokenizer.tokenize_to_ids("irrelevant")

        self.assertEqual(ids, [1, 2, 0, 4, 5, 6])
        self.assertEqual(tokenizer.unk_token_id, 3)
        self.assertEqual(tokenizer.get_vocab_size(), 7)
        self.assertEqual(tokenizer.get_pad_token_id(), 6)

    def test_decode_renders_restricted_unknown_and_special_ids(self) -> None:
        from src.dpnlp_pipeline.tokenizer import RestrictedTokenizer

        class FakeBackendTokenizer:
            vocab_size = 6
            eos_token = "<eos>"
            bos_token = "<bos>"
            pad_token = "<pad>"
            eos_token_id = 4
            bos_token_id = 3
            pad_token_id = 5
            special_tokens_map = {"eos_token": "<eos>", "bos_token": "<bos>", "pad_token": "<pad>"}

            def decode(self, input_ids):
                pieces = {
                    0: "zero",
                    1: " one",
                    2: " two",
                    3: "<bos>",
                    4: "<eos>",
                    5: "<pad>",
                }
                return "".join(pieces[token_id] for token_id in input_ids)

        tokenizer = RestrictedTokenizer(FakeBackendTokenizer(), kept_token_ids=[2, 0, 1])

        self.assertEqual(tokenizer.decode([0, 1, 2]), " twozero one")
        self.assertEqual(tokenizer.decode([tokenizer.unk_token_id]), "<unk>")
        self.assertEqual(tokenizer.decode([4, 5, 6]), "<bos><eos><pad>")


class CorpusVocabTests(unittest.TestCase):
    def test_build_vocab_payload_keeps_most_common_non_special_tokens(self) -> None:
        from src.dpnlp_pipeline.tokenizer import build_restricted_vocab_payload

        class FakeBackendTokenizer:
            eos_token_id = 99
            bos_token_id = None
            pad_token_id = None

            def __call__(self, text, truncation=False, return_attention_mask=False, return_tensors=None):
                mapping = {
                    "alpha<|endoftext|>": [2, 2, 99],
                    "beta<|endoftext|>": [1, 2, 99],
                    "gamma<|endoftext|>": [1, 3, 99],
                }
                return {"input_ids": mapping[text]}

        with TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "TinyStories-train.txt"
            train_path.write_text("alpha<|endoftext|>beta<|endoftext|>gamma<|endoftext|>")

            payload = build_restricted_vocab_payload(train_path, FakeBackendTokenizer(), pretrained_name="fake/gpt-neo", vocab_size=2)

        self.assertEqual(payload["pretrained_name"], "fake/gpt-neo")
        self.assertEqual(payload["vocab_size"], 2)
        self.assertEqual(payload["kept_token_ids"], [2, 1])


class TrainingPreviewTests(unittest.TestCase):
    def test_maybe_print_training_preview_only_runs_on_interval(self) -> None:
        from src.dpnlp_pipeline import training

        class FakeModel:
            def __init__(self) -> None:
                self.training = True
                self.eval_calls = 0
                self.train_calls = 0

            def eval(self):
                self.training = False
                self.eval_calls += 1
                return self

            def train(self):
                self.training = True
                self.train_calls += 1
                return self

        model = FakeModel()
        tokenizer = object()

        with mock.patch.object(training, "generate_completion", return_value="preview text") as generate_completion, mock.patch(
            "builtins.print"
        ) as print_mock:
            training.maybe_print_training_preview(model, tokenizer, step=999, device="cpu")
            generate_completion.assert_not_called()
            print_mock.assert_not_called()

            training.maybe_print_training_preview(model, tokenizer, step=1000, device="cpu")

        generate_completion.assert_called_once_with(
            model,
            tokenizer,
            training.TRAINING_PREVIEW_PROMPT,
            max_new_tokens=training.TRAINING_PREVIEW_MAX_NEW_TOKENS,
            temperature=0.0,
            device="cpu",
        )
        print_mock.assert_called_once_with("[batch 1000] preview text")
        self.assertEqual(model.eval_calls, 1)
        self.assertEqual(model.train_calls, 1)
        self.assertTrue(model.training)


class ReportingTests(unittest.TestCase):
    def test_report_summarizes_claim_support(self) -> None:
        from src.dpnlp_pipeline.reporting import build_claim_summary

        evaluation_rows = [
            {"model_size": "xs", "grammar": 6.5, "creativity": 2.5, "consistency": 2.0, "max_overlap": 0.09},
            {"model_size": "small", "grammar": 7.0, "creativity": 3.5, "consistency": 3.0, "max_overlap": 0.08},
            {"model_size": "medium", "grammar": 7.5, "creativity": 5.0, "consistency": 5.2, "max_overlap": 0.07},
            {"model_size": "large", "grammar": 8.1, "creativity": 6.2, "consistency": 6.4, "max_overlap": 0.06},
            {"model_size": "xl", "grammar": 8.4, "creativity": 6.8, "consistency": 7.0, "max_overlap": 0.05},
        ]

        summary = build_claim_summary(evaluation_rows)

        self.assertEqual(summary["larger_models_more_coherent"], "supported")
        self.assertEqual(summary["small_models_grammar_before_creativity_consistency"], "supported")
        self.assertEqual(summary["outputs_not_simple_memorization"], "supported")

    def test_report_sorts_rows_by_model_size_instead_of_input_order(self) -> None:
        from src.dpnlp_pipeline.reporting import build_claim_summary

        evaluation_rows = [
            {"model_size": "large", "grammar": 8.1, "creativity": 6.2, "consistency": 6.4, "max_overlap": 0.06},
            {"model_size": "xs", "grammar": 6.5, "creativity": 2.5, "consistency": 2.0, "max_overlap": 0.09},
            {"model_size": "xl", "grammar": 8.4, "creativity": 6.8, "consistency": 7.0, "max_overlap": 0.05},
            {"model_size": "medium", "grammar": 7.5, "creativity": 5.0, "consistency": 5.2, "max_overlap": 0.07},
            {"model_size": "small", "grammar": 7.0, "creativity": 3.5, "consistency": 3.0, "max_overlap": 0.08},
        ]

        summary = build_claim_summary(evaluation_rows)

        self.assertEqual(summary["larger_models_more_coherent"], "supported")
        self.assertEqual(summary["small_models_grammar_before_creativity_consistency"], "supported")


class PromptTests(unittest.TestCase):
    def test_default_prompt_set_has_paper_sized_count(self) -> None:
        from src.dpnlp_pipeline.prompts import DEFAULT_GPT_EVAL_PROMPTS

        self.assertEqual(len(DEFAULT_GPT_EVAL_PROMPTS), 50)
        self.assertTrue(all(prompt.endswith(" ") or prompt.endswith('"') for prompt in DEFAULT_GPT_EVAL_PROMPTS))

    def test_overlap_metric_detects_shared_ngrams(self) -> None:
        from src.dpnlp_pipeline.evaluation import ngram_overlap

        score = ngram_overlap("the cat sat on the mat", "cat sat on the mat and purred", n=3)
        self.assertGreater(score, 0.0)

    def test_overlap_metric_can_reuse_precomputed_reference_counts(self) -> None:
        from src.dpnlp_pipeline.evaluation import build_ngram_counter, ngram_overlap, ngram_overlap_against_counts

        reference = "the cat sat on the mat"
        candidate = "cat sat on the mat and purred"

        reference_counts = build_ngram_counter(reference, n=3)

        self.assertEqual(ngram_overlap(reference, candidate, n=3), ngram_overlap_against_counts(reference_counts, candidate, n=3))

    def test_evaluate_overlap_only_computes_train_and_valid_once_per_row(self) -> None:
        import sys

        fake_matplotlib = ModuleType("matplotlib")
        fake_pyplot = ModuleType("matplotlib.pyplot")
        fake_matplotlib.pyplot = fake_pyplot
        fake_generation = ModuleType("src.dpnlp_pipeline.generation")
        fake_generation.generate_completion = lambda *args, **kwargs: None
        fake_gpt_eval = ModuleType("src.dpnlp_pipeline.gpt_eval")
        fake_gpt_eval.grade_completion = lambda *args, **kwargs: {}
        fake_io = ModuleType("src.dpnlp_pipeline.io")
        fake_io.load_checkpoint = lambda *args, **kwargs: {}
        fake_io.load_json = lambda *args, **kwargs: {}
        fake_io.save_checkpoint = lambda *args, **kwargs: None
        fake_io.save_json = lambda *args, **kwargs: None
        fake_modeling = ModuleType("src.dpnlp_pipeline.modeling")
        fake_modeling.build_model = lambda *args, **kwargs: None
        fake_prompts = ModuleType("src.dpnlp_pipeline.prompts")
        fake_prompts.DEFAULT_GPT_EVAL_PROMPTS = []
        fake_reporting = ModuleType("src.dpnlp_pipeline.reporting")
        fake_reporting.build_claim_summary = lambda *args, **kwargs: {}
        fake_tokenizer = ModuleType("src.dpnlp_pipeline.tokenizer")
        fake_tokenizer.TokenizerSpec = object
        fake_tokenizer.build_tokenizer = lambda *args, **kwargs: None
        fake_training = ModuleType("src.dpnlp_pipeline.training")
        fake_training.train_model = lambda *args, **kwargs: None

        with mock.patch.dict(
            sys.modules,
            {
                "matplotlib": fake_matplotlib,
                "matplotlib.pyplot": fake_pyplot,
                "src.dpnlp_pipeline.generation": fake_generation,
                "src.dpnlp_pipeline.gpt_eval": fake_gpt_eval,
                "src.dpnlp_pipeline.io": fake_io,
                "src.dpnlp_pipeline.modeling": fake_modeling,
                "src.dpnlp_pipeline.prompts": fake_prompts,
                "src.dpnlp_pipeline.reporting": fake_reporting,
                "src.dpnlp_pipeline.tokenizer": fake_tokenizer,
                "src.dpnlp_pipeline.training": fake_training,
            },
        ):
            from src.dpnlp_pipeline import pipeline

            outputs = [{"model_size": "small", "prompt": "Prompt", "completion": "completion text"}]

            class FakeTextFile:
                def __init__(self, text: str) -> None:
                    self.text = text
                    self.encoding = None

                def read_text(self, encoding=None) -> str:
                    self.encoding = encoding
                    return self.text

            class FakePaths:
                train_path = FakeTextFile("train text")
                valid_path = FakeTextFile("valid text")

            with mock.patch.object(pipeline, "resolve_paths", return_value=FakePaths()), mock.patch.object(
                pipeline, "ngram_overlap_against_counts", return_value=0.25
            ) as overlap_against_counts:
                pipeline.evaluate_overlap(outputs, repo_root=Path.cwd())

        self.assertEqual(overlap_against_counts.call_count, 2)
        self.assertEqual(FakePaths.train_path.encoding, "utf-8")
        self.assertEqual(FakePaths.valid_path.encoding, "utf-8")

    def test_pipeline_can_describe_itself(self) -> None:
        import json
        import subprocess

        command = ["conda", "run", "-n", "chatbot", "python", "-m", "src.dpnlp_pipeline", "describe"]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(completed.stdout)

        self.assertEqual(payload["defaults"]["context_length"], 512)
        self.assertEqual(payload["model_specs"]["small"]["target_params"], 3000000)

    def test_write_prompts_command_creates_artifact(self) -> None:
        import subprocess

        command = ["conda", "run", "-n", "chatbot", "python", "-m", "src.dpnlp_pipeline", "write-prompts"]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)

        prompt_path = Path(completed.stdout.strip())
        self.assertTrue(prompt_path.exists())

    def test_generate_for_prompt_set_runs_all_prompts_for_one_model(self) -> None:
        from src.dpnlp_pipeline import pipeline

        prompts = ["Prompt A", "Prompt B", "Prompt C"]

        with mock.patch.object(pipeline, "load_prompt_set", return_value=prompts), mock.patch.object(
            pipeline,
            "generate_from_checkpoint",
            side_effect=lambda model_size, prompt, **kwargs: {"model_size": model_size, "prompt": prompt, "completion": f"{model_size}:{prompt}"},
        ):
            rows = pipeline.generate_for_prompt_set("small", repo_root=Path.cwd())

        self.assertEqual(len(rows), 3)
        self.assertEqual([row["prompt"] for row in rows], prompts)
        self.assertTrue(all(row["model_size"] == "small" for row in rows))

    def test_generate_all_for_prompt_set_runs_all_models(self) -> None:
        from src.dpnlp_pipeline import pipeline

        prompts = ["Prompt A", "Prompt B"]

        with mock.patch.object(pipeline, "load_prompt_set", return_value=prompts), mock.patch.object(
            pipeline,
            "generate_from_checkpoint",
            side_effect=lambda model_size, prompt, **kwargs: {"model_size": model_size, "prompt": prompt, "completion": f"{model_size}:{prompt}"},
        ):
            rows = pipeline.generate_all_for_prompt_set(repo_root=Path.cwd())

        expected_models = list(pipeline.MODEL_SPECS.keys())
        self.assertEqual(len(rows), len(expected_models) * len(prompts))
        self.assertEqual(sorted({row["model_size"] for row in rows}), sorted(expected_models))

    def test_write_loss_plot_creates_png_for_model(self) -> None:
        from src.dpnlp_pipeline import pipeline

        metrics_path = Path("artifacts/dpnlp_pipeline/reports/test_model_train_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text('{"losses":[3.0,2.0,1.0],"model_spec":{"name":"test_model"}}')

        with mock.patch.object(pipeline, "resolve_paths") as resolve_paths:
            class FakePaths:
                reports_dir = Path("artifacts/dpnlp_pipeline/reports")

            resolve_paths.return_value = FakePaths()
            output_path = pipeline.write_loss_plot("test_model", repo_root=Path.cwd())

        self.assertTrue(Path(output_path).exists())
        self.assertEqual(Path(output_path).suffix, ".png")

    def test_write_loss_plot_supports_log_scale(self) -> None:
        from src.dpnlp_pipeline import pipeline

        metrics_path = Path("artifacts/dpnlp_pipeline/reports/test_model_log_train_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text('{"losses":[1000.0,100.0,10.0],"model_spec":{"name":"test_model_log"}}')

        with mock.patch.object(pipeline, "resolve_paths") as resolve_paths:
            class FakePaths:
                reports_dir = Path("artifacts/dpnlp_pipeline/reports")

            resolve_paths.return_value = FakePaths()
            output_path = pipeline.write_loss_plot("test_model_log", repo_root=Path.cwd(), log_scale=True)

        self.assertTrue(Path(output_path).exists())
        self.assertIn("log", Path(output_path).stem)


class GptEvalTests(unittest.TestCase):
    def test_gpt_eval_prompt_uses_paper_appendix_wording(self) -> None:
        from src.dpnlp_pipeline.gpt_eval import build_gpt_eval_prompt

        prompt = build_gpt_eval_prompt("Beginning of story", "Ending of story")

        self.assertIn("The following exercise, the student is given a beginning of a story.", prompt)
        self.assertIn("The symbol *** marks the separator", prompt)
        self.assertIn("Please provide your general assessment", prompt)
        self.assertIn("whether the plot makes sense", prompt)
        self.assertIn("what the age of the student might be", prompt)
        self.assertIn("***", prompt)

    def test_grade_completion_parses_extended_response_fields(self) -> None:
        from src.dpnlp_pipeline.gpt_eval import grade_completion

        class FakeResponse:
            output_text = (
                '{"general_assessment":"Coherent and simple.",'
                '"grammar":7,'
                '"creativity":6,'
                '"consistency":8,'
                '"plot_sense":7,'
                '"age_guess":"6-8"}'
            )

        class FakeResponses:
            def create(self, **kwargs):
                return FakeResponse()

        class FakeClient:
            responses = FakeResponses()

        scores = grade_completion("Once upon a time", "there was a cat", client=FakeClient(), model="fake")

        self.assertEqual(scores["general_assessment"], "Coherent and simple.")
        self.assertEqual(scores["grammar"], 7.0)
        self.assertEqual(scores["creativity"], 6.0)
        self.assertEqual(scores["consistency"], 8.0)
        self.assertEqual(scores["plot_sense"], 7.0)
        self.assertEqual(scores["age_guess"], "6-8")


class ParameterCountingTests(unittest.TestCase):
    def test_transformer_param_number_matches_unique_parameter_total(self) -> None:
        from src.transformer.transformer import Transformer

        model = Transformer(vocab_size=257, d_model=128, n_layers=2, d_ff=512, n_heads=8)
        unique_total = sum(parameter.numel() for parameter in model.parameters())

        self.assertEqual(model.get_param_number(), unique_total)

    def test_decoder_block_param_number_matches_unique_parameter_total_with_bias(self) -> None:
        from src.transformer.utils.decoder_block import DecoderBlock

        block = DecoderBlock(depth=0, d_model=128, d_ff=384, n_heads=8, bias=True)
        unique_total = sum(parameter.numel() for parameter in block.parameters())

        self.assertEqual(block.get_param_number(), unique_total)


class TrainingCheckpointTests(unittest.TestCase):
    def test_training_checkpoint_round_trip_includes_optimizer_state(self) -> None:
        import tempfile
        import torch
        from torch import nn
        from torch.optim.adamw import AdamW

        from src.dpnlp_pipeline.io import load_training_checkpoint, save_training_checkpoint

        model = nn.Linear(2, 1)
        optimizer = AdamW(model.parameters(), lr=0.1)
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "training_state.pt"
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                metadata={"global_step": 7, "losses": [1.5, 1.25]},
            )

            payload = load_training_checkpoint(checkpoint_path)

        self.assertEqual(payload["metadata"]["global_step"], 7)
        self.assertEqual(payload["metadata"]["losses"], [1.5, 1.25])
        self.assertIn("state_dict", payload)
        self.assertIn("optimizer_state_dict", payload)
        self.assertTrue(payload["optimizer_state_dict"]["state"])

    def test_training_checkpoint_save_keeps_temp_file_when_replace_is_blocked(self) -> None:
        import tempfile
        import torch
        from torch import nn
        from torch.optim.adamw import AdamW

        from src.dpnlp_pipeline.io import save_training_checkpoint

        model = nn.Linear(2, 1)
        optimizer = AdamW(model.parameters(), lr=0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "training_state.pt"

            with mock.patch.object(Path, "replace", side_effect=PermissionError("locked by Windows")):
                save_training_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    metadata={"global_step": 1, "losses": [1.5]},
                )

            self.assertTrue(Path(str(checkpoint_path) + ".tmp").exists())

    def test_training_checkpoint_load_prefers_newer_temp_checkpoint(self) -> None:
        import tempfile
        import torch
        from torch import nn
        from torch.optim.adamw import AdamW

        from src.dpnlp_pipeline.io import load_training_checkpoint, save_training_checkpoint

        model = nn.Linear(2, 1)
        optimizer = AdamW(model.parameters(), lr=0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "training_state.pt"
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                metadata={"global_step": 10, "losses": [1.5]},
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metadata": {"global_step": 11, "losses": [1.5, 1.25]},
                },
                Path(str(checkpoint_path) + ".tmp"),
            )

            payload = load_training_checkpoint(checkpoint_path)

        self.assertEqual(payload["metadata"]["global_step"], 11)

    def test_train_model_resumes_from_periodic_checkpoint(self) -> None:
        import tempfile
        import torch
        from torch import nn
        from torch.utils.data import Dataset

        from src.dpnlp_pipeline.config import ModelSpec
        from src.dpnlp_pipeline.training import train_model

        class FakeTokenizer:
            def get_vocab_size(self):
                return 5

        class FakeDataset(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, -100])

        class TinyLm(nn.Module):
            def __init__(self):
                super().__init__()
                self.class_zero_logit = nn.Parameter(torch.tensor(0.0))

            def forward(self, input_ids):
                logits = torch.zeros(input_ids.size(0), input_ids.size(1), 5)
                logits[:, :, 0] = self.class_zero_logit
                return logits

            def get_param_number(self):
                return sum(parameter.numel() for parameter in self.parameters())

        defaults = SimpleNamespace(context_length=4, train_epochs=1)
        model_spec = ModelSpec("tiny", 1, d_model=1, n_layers=1, n_heads=1, d_ff=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "tiny_training_state.pt"

            with mock.patch("src.dpnlp_pipeline.training.build_dataset", return_value=FakeDataset()), mock.patch(
                "src.dpnlp_pipeline.training.build_model", side_effect=lambda *args: TinyLm()
            ):
                first_run = train_model(
                    model_spec,
                    FakeTokenizer(),
                    defaults,
                    paths=SimpleNamespace(train_path=Path("unused")),
                    batch_size=1,
                    max_steps=2,
                    device="cpu",
                    checkpoint_path=checkpoint_path,
                    save_every_steps=1,
                )
                resumed = train_model(
                    model_spec,
                    FakeTokenizer(),
                    defaults,
                    paths=SimpleNamespace(train_path=Path("unused")),
                    batch_size=1,
                    max_steps=3,
                    device="cpu",
                    checkpoint_path=checkpoint_path,
                    resume_from_checkpoint=checkpoint_path,
                    save_every_steps=1,
                )

        self.assertEqual(first_run.global_step, 2)
        self.assertEqual(resumed.global_step, 3)
        self.assertEqual(len(resumed.train_loss_history), 3)


if __name__ == "__main__":
    unittest.main()
