import pandas as pd
import json
import uuid
df=pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\DMeX\Dmex_Rag\dmex_docs.csv")
df.dropna(subset='Text')
chunks=[]
for url,group in df.groupby("URL"):
    current_section=""
    current_subsection=""
    current_chunk=[]
    for _ ,row in group.iterrows():
        tag=row["Tag"].lower()
        text=row["Text"].strip()
        if tag in ["h2","h3"]:
            if current_chunk:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text":" ".join(current_chunk),
                    "metadata":{
                        "source":url,
                        "section":current_section,
                        "tags":["li","p"]
                    }
                })
                current_chunk=[]
            current_section=text  
        elif tag in ["li", "p", "h4", "td"]:
            current_chunk.append(text)
    if current_chunk:
        chunks.append({
            "id": str(uuid.uuid4()),
            "text": " ".join(current_chunk),
            "metadata": {
                "source": url,
                "section": current_section,
                "tags": ["li", "p"]
            }
        })        
with open("cleaned_rag_dataset.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)
