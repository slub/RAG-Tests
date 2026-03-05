---
tags:
  - project
  - llm
  - ai
  - rag
concluded:
description: LLM-basierter Informationsabruf aus SLUB Website, Intranet und Qucosa mittels Retrieval Augmented Generation
state: in progress
from_obsidian: true
---
# RAG Tests, SLUB

Im Projekt sollen die erweiterten Möglichkeiten die LLMs im Bereich der Informationsaufbereitung bieten anhand dreier SLUB-relevanter Szenarien sowohl technisch als auch organisatorisch bzw. organisationell exploriert werden. Mit Hilfe der Retrieval Augmented Generation sollen spezialisierte LLMs für 
0. RAG Testsystem
1. die SLUB Webseite, 
2. das SLUB Open Access Repositorium Qucosa und
3. das SLUB Intranet
konstruiert werden und in ihrer Abfragequalität bezüglich der allgemeinen Qualität, der Nützlichkeit für potenzielle Anwender:innen und des Ressourcenbedarfes evaluiert werden. Im Projekt sind inbesondere Herausforderungen durch organisatorischen Rahmenbedingungen wie eine sich stetig verändernde Datengrundlage sowie Aspekte des Datenschutzes zu berücksichtigen. Für die SLUB entsteht dabei mindestens Erfahrung und Bewertungskompetenz bezüglich technischer Möglichkeiten durch den Einsatz lokaler, spezialisierter LLMs und im besten Fall ein wertvolles Recherchewerkzeug.

Similar to:
https://github.com/jmiba/ai-service-chatbot

## Anleitungen

- RAG auf dem Laptop (Linux Mint) einrichten: [RAG, Lokal](notes/rag__lokal.md)
- RAG auf TUD research cloud [RAG, VM](notes/rag__vm.md)

## Test-Repo

- https://github.com/rue-a/rag-test-system

## Probleme

- [RAG, Unvollsändiger Retrival](rag__unvollsändiger_retrival.md): Abfrage an Vectorstore liefert unvollständige Antwort
