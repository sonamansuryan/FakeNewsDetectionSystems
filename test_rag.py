import sys
import os
import torch

sys.path.append(os.getcwd())

from src.models.roberta_model import RoBERTaModel
from src.data_collection.rag_retriever import RAGRetriever
from src.counter_narrative.generator import CounterNarrativeGenerator

print("Initializing RoBERTa...")
roberta = RoBERTaModel(config_path="configs/config.yaml")

roberta.model_settings['name'] = "models/roberta/acc_8183/checkpoint-5100"

roberta.load_model()

retriever = RAGRetriever()
generator = CounterNarrativeGenerator(model_name="mistral", timeout=120)

test_claims = [
    "Drinking alkaline water can cure Stage 1 lung cancer by neutralizing the body's pH levels according to recent holistic studies",

    "The FDA has officially approved the first CRISPR-based gene therapy for the treatment of sickle cell disease in December 2023",

    "Germany has officially announced a new 'Citizen Score' law to restrict travel for individuals based on their social media behavior starting July 2025",

    "Sweden has officially become the 32nd member of NATO in March 2024, ending decades of military neutrality"
]

print(f"\n{'=' * 70}")
print(f"STARTING REAL-TIME SYSTEM (REAL ROBERTA 83%+ ACCURACY)")
print(f"{'=' * 70}")

for claim in test_claims:
    print(f"\n[CLAIM]: {claim}")

    print("RoBERTa is analyzing...")
    roberta_result = roberta.predict(claim)

    print(f"Result: {roberta_result.label} ({roberta_result.confidence:.2%})")

    print("RAG is searching context...")
    context = retriever.retrieve(claim)

    print("Mistral is generating explanation...")
    final_result = generator.generate(claim, roberta_result, context)

    print("\n" + "-" * 50)
    print(f"CLAIM: {claim}")
    print(f"ROBERTA INITIAL: {final_result.roberta_label} ({final_result.roberta_confidence:.2%})")

    print(f"FINAL VERDICT: {final_result.final_label}")
    if final_result.is_correction:
        print("⚠️  NOTICE: Mistral overrode RoBERTa's decision based on RAG evidence!")

    print(f"\nMISTRAL EXPLANATION:\n{final_result.explanation}")
    print(f"\nCOUNTER-NARRATIVE:\n{final_result.counter_narrative}")
    print("-" * 50)

print(f"\n{'=' * 70}")
print(f"FULL SYSTEM TEST COMPLETE")
print(f"{'=' * 70}")