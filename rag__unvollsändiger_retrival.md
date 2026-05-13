---
tags:
  - notes
  - rag
  - problem
from_obsidian: true
---

Als Beispiel wurde das HTML des [SLUB Veranstaltungskalenders](https://www.slub-dresden.de/besuchen/veranstaltungen) (von Ende Februar) in die Knowledge Base (KB) eingespielt. Anfragen nach Veranstaltungen an bestimmten Tagen scheitern, die Antworten sind nicht vollständig. 

**Mögliche Fehlerquellen:** 
- Die korrekten Chunks werden nicht retrieved. Auf Basis der Frage, werden die Chunks im Vectorstore auf Relevanz gerankend. Aus den k-relevantesten Chunks wird die Antwort generiert. Wenn die gesuchten Antworten nicht in diesen relevantesten Chunks sind, scheitert die Anfrage.
- Die korrekten Chunks wurden retrieved, aber die richtige Antwort gelangt nicht in den *Kontext*. -> Problem in der Consolidation Strategy


**Am wahrscheinlichsten entsteht das Problem durch ungünstiges Chunking.** Teste andere Chunker.

__Lösung__: Unstructured ignoriert manche HTML tags, z.B. `article` und erzeugt einzelne Chunks aus den Unterelementen, obwohl sie semantisch/strukturell zusammenhängen. Preprocessing Funktion implementiert, die das Problem löst.


%% 

## LlamaIndex

[Retrieval Augmented Generation - Examples & How to Build One | Mirascope](https://mirascope.com/blog/rag-llm-example)

- Der Chunking Algo scheint proprietär zu sein (llama pase)

## Haystack

[Creating Your First QA Pipeline with Retrieval-Augmentation | Haystack](https://haystack.deepset.ai/tutorials/27_first_rag_pipeline)

`load_dataset` method

## DSPy


## RAGFlow


## LightRAG

## LLMWare %%
