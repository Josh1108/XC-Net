import spacy
model="en_core_web_sm"

def pos_tag_job(job):
  nlp = spacy.load(model)
  job1=nlp(job)
  return job1

def remove_verb(job):
  pos_job = pos_tag_job(job)
  job_tok =[]
  for token in pos_job:
    if str(token.pos_) == "VERB":
      continue
    else:
      job_tok.append(token.text)
  jd = " ".join(job_tok)
  return jd

# job= "requirements preferably least 1 year relevant exp providing support director admin manager assisting daily day day activities carry indoor sales network tele marketing exp willing learn also generate quotation invoicing etc sales coordination functions"

# print(remove_verb(job))