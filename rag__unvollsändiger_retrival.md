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
