# from transformers import AutoTokenizer

# from transformers import T5Tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig
import torch
# from transformers.convert_slow_tokenizer import convert_slow_tokenizer

original_tokenizer = T5Tokenizer.from_pretrained('sentencepiece.bpe.model')
mt5_tokenizer = T5Tokenizer.from_pretrained('google/mt5-base')
# print(len(original_tokenizer))

# fast_tokenizer = convert_slow_tokenizer(original_tokenizer)
# fast_tokenizer.save('spm_tokenizer.json')
# print(fast_tokenizer)

# print(original_tokenizer.eos_token_id)
# print(original_tokenizer.pad_token_id)
# print(original_tokenizer.cls_token_id)


# print(original_tokenizer(['สวัสดีทุกๆคน', 'testing english language'], padding = True))

# config = get_config(args)

config = AutoConfig.from_pretrained(
    'google/t5-v1_1-base',
)
config.dropout_rate = 0.0
config.vocab_size = len(original_tokenizer)
model = T5ForConditionalGeneration(
    config,
)
model.load_state_dict(torch.load('logs/2023-03-24/19-01-44/checkpoint-pt-60000/pytorch_model.bin'))

mt5_model = T5ForConditionalGeneration.from_pretrained('google/mt5-base')
# input_text = """ขอต้อนรับสู่เว็บไซต์ ThaiGPT ครับ 

# ThaiGPT เป็นบริษัทที่ต่อตั้งขึ้นในเดือนกุมภาพันธ์ 2023 โดย นายแพทย์ภาณุทัต เตชะเสน และคุณโดม เจริญยศ โดยตั้งเป้าให้เป็นบริษัทด้านวิจัยและพัฒนาเกี่ยวกับ AI โดยเฉพาะอย่างยิ่งในด้าน Large Language Model และ Generative AI 

# ปัจจุบันการใช้งานด้าน NLP และ Generative Model แพร่หลายขึ้นอย่างรวดเร็วทั่วโลก อุปสรรคอย่างหนึ่งของประเทศไทยในการประยุกต์ใช้เทคโนโลยีทางด้านนี้คือภาษาไทย ดังนั้นการวิจัยและพัฒนาเกี่ยวกับภาษาไทยในด้านนี้จึงมีความจำเป็นอย่างเร่งด่วน
# """.lower()
# input_text = "เว็บไซต์ข่าวของจีนรายงานว่า ในพิธีเปิดตัว ซูซาโนได้รับรางวัลกลุ่มนวัตกรรมแบบเปิด จากรัฐบาลของเขตใหม่ผู่ตง และมีการลงนามข้อตกลงความร่วมมือเชิงกลยุทธ์เกี่ยวกับการวิจัยและพัฒนาวัสดุชีวภาพกับสถาบันวิจัยและมหาวิทยาลัยหลายแห่งในประเทศจีน รวมถึงร่วมกันเปิดตัวโครงการพัฒนาอย่างยั่งยืนกับลูกค้าและซัพพลายเออร์ท้องถิ่นในจีน"
# input_text = "ปักกิ่ง, 23 มี.ค. (ซินหัว) -- เมื่อวันอังคาร (21 มี.ค.) ที่ผ่านมา ซูซาโน (Suzano) บริษัทสัญชาติบราซิล ผู้ผลิตเยื่อกระดาษรายใหญ่ของโลก ได้ก่อตั้งศูนย์นวัตกรรมประจำประเทศจีนในนครเซี่ยงไฮ้ ซึ่งมีดีกรีเป็นฐานวิจัยและพัฒนา รวมถึงเป็นศูนย์นวัตกรรมแห่งแรกในเอเชียของบริษัท"
input_text = """หลังพัฒนาขึ้นสู่ประเทศที่มีความเจริญก้าวหน้าลำดับต้น ๆ ของโลกทั้งด้านเศรษฐกิจ เทคโนโลยี และอุตสาหกรรมบันเทิง ประเด็นต่าง ๆ ที่เกี่ยวกับเกาหลีใต้ก็ทวีความสนใจ และน่าติดตาม
.
ซึ่งความสำเร็จแบบก้าวกระโดดช่วงไม่ถึงหนึ่งอายุคนเช่นนี้ จะเกิดขึ้นไม่ได้เลย หากประชาชนไม่เป็นฟันเฟืองช่วยผลักดันและขับเคลื่อน
.
นี่จึงทำให้เมื่อมองผ่านเลนส์ด้านทรัพยากรมนุษย์แล้ว เกาหลีใต้ เป็นประเทศประชากรเปี่ยมคุณภาพ เต็มไปด้วยการทุ่มเท และเป็นองค์ประกอบสำคัญในการยกระดับประเทศ"""
print('input_text:', input_text)

input_ids = original_tokenizer(input_text, return_tensors="pt").input_ids
# print(input_ids)
outputs = model.generate(input_ids, max_length = 20, repetition_penalty=2.0)
# print(outputs)
output_text = original_tokenizer.decode(outputs[0], skip_special_tokens=True)

print('\noutput_text t5:', output_text)

# print('\ninput_text mt5:', input_text)

input_ids = mt5_tokenizer(input_text, return_tensors="pt").input_ids
# print(input_ids)
outputs = mt5_model.generate(input_ids, max_length = 20, repetition_penalty=2.0)
# print(outputs)
output_text = mt5_tokenizer.decode(outputs[0], skip_special_tokens=True)

print('\noutput_text mt5:', output_text)
