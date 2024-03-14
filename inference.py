import torch
import time
import os
import json
import tqdm
from typing import Optional 
from pathlib import Path
from sentencepiece import SentencePieceProcessor
from model import ModelArgs, Transformer

class LLaMA:
    def __init__(
        self, 
        model: Transformer,
        tokenizer: SentencePieceProcessor,
        model_args: ModelArgs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model.args

    @staticmethod
    def build(
        checkpoint_dir: str,
        load_model: bool,
        tokenizer_path: str,
        max_batch_size: int,
        max_seg_len: int,
        device: str
    ):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoint_dir).glob('*.pth'))
            assert len(checkpoints) > 0, 'No checkpoint path'
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint {ckpt_path}')
            checkpoint = torch.load(checkpoints)
            print(f'Loaded checkpoint in {time.time() - prev_time}')
        with open(os.path.join(checkpoints, 'params.json'), 'r') as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_batch_size = max_batch_size,
            max_seg_len = max_seg_len,
            device = device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device=='cuda':
            torch.set_default_tensor_type(torch.cuda.HaftTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)

        return LLaMA(model, tokenizer, model_args)
        
    def text_completion(
        self,
        prompts: list[str],
        max_gen_len: Optional[int],
        temperature: Optional[float] = None,
        top_p: Optional[float] = 0.9
    ):
        # Max generated text lenth 
        if max_gen_len is None:
            max_gen_len = self.args.max_seg_len

        # Encode prompt
        prompt_tokens = [
            self.tokenzer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
            for prompt in prompts
        ]

        batch_size = len(prompt_tokens)
        assert batch_size < self.args.max_batch_size, f'Batch size must be smaller than {self.args.max_batch_size}'

        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len < self.args.max_seg_len, f'Max sequence langth must be smaller {self.args.max_seg_len}'

        total_len = min(self.args.max_seg_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id

        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.args.device)
        for i, prompt in enumerate(prompt_tokens):
            tokens[i, :len(prompt)] = torch.Tensor(prompt, dtype=torch.long, device=args.device)

        eos_reached = torch.tensor([False] * batch_size, device=self.args.device)
        prompt_mask = tokens != pad_id
        cur_iteration = tqdm(range(1, total_len), desc='Generating tokens')
        for cur_pos in cur_iteration:
            with torch.no_grad():
                # (Batch, 1, vocab_size)
                logits = self.model(tokens[:, cur_pos-1:cur_pos], cur_pos)
            if temperature > 0:
                # (Batch, 1, vocab_size)
                probs = torch.softmax(logits[:, -1]/temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            
            next_token = next_token.reshape(-1)
            next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            eos_reached |= (~prompt_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            if all(eos_reached):
                break
        
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens):
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        
        return out_tokens, out_text

    def sample_top_p(probs: torch.Tensor, p: float):
        sorted_probs, probs_index = torch.sort(probs, dim=-1, descending=True)
        sorted_probs_sum = torch.cumsum(sorted_probs, dim=-1)        
        shifted_probs = sorted_probs_sum - sorted_probs
        probs_mask = shifted_probs > p
        sorted_probs[probs_mask] = 0
        sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
            
        sorted_probs_idx = torch.multinomial(sorted_probs,num_samples=1)
        next_token = torch.gather(sorted_probs, -1, sorted_probs_idx)
        return next_token



if '__name__' == '__main__':
    torch.manual_seed(0)

    allow_cuda = True
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot promt
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Umar Jamil
        Decision: 
        """
    ]

    model = LLaMA.build(
        checkpoints_dir='llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )

    out_tokens, out_texts = model.text_completion(        
        prompts = prompts,
        max_gen_len = 64,
        temperature = 0.6,
        top_p = 0.9
    )

    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)