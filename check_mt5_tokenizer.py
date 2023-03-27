from transformers import AutoTokenizer

from transformers import T5Tokenizer
from transformers.convert_slow_tokenizer import convert_slow_tokenizer

original_tokenizer = T5Tokenizer.from_pretrained('sentencepiece.bpe.model')
print(len(original_tokenizer))

fast_tokenizer = convert_slow_tokenizer(original_tokenizer)
fast_tokenizer.save('spm_tokenizer.json')
print(fast_tokenizer)

print(original_tokenizer.eos_token_id)
print(original_tokenizer.pad_token_id)
print(original_tokenizer.cls_token_id)


print(original_tokenizer(['สวัสดีทุกๆคน', 'testing english language'], padding = True))
print(original_tokenizer.decode(original_tokenizer.encode('สวัสดีทุกๆคน \n\n {} sdfd   sdffds testing english language')))
print(len(original_tokenizer.tokenize('\n\n {} So, it gets encoded to ??. SentencePiece has a byte fallback feature but it was not available when we trained our sentencepiece model.'.lower())), 'len')

    
tokenizer = AutoTokenizer.from_pretrained(
        'airesearch/wangchanberta-base-att-spm-uncased',
        use_fast=True,
        # cls_token = None
)
# tokenizer.

print(tokenizer.eos_token_id)
print(tokenizer.pad_token_id)
print(tokenizer.cls_token_id)


print(tokenizer(['testing english language', 'test'], padding = True))


tokenizer2 = AutoTokenizer.from_pretrained(
        # 'google/t5-v1_1-base',
        'google/mt5-base',
        use_fast=True
    )

print(tokenizer2.eos_token_id)
print(tokenizer2.pad_token_id)
print(len(tokenizer2.tokenize('\n\n {} So, it gets encoded to ??. SentencePiece has a byte fallback feature but it was not available when we trained our sentencepiece model.'.lower())), 'len')

print(tokenizer2(['testing english language', 'test'], padding = True))

