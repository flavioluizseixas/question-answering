from haystack.telemetry import tutorial_running
tutorial_running(27)

from haystack.document_stores.in_memory import InMemoryDocumentStore
document_store = InMemoryDocumentStore()

from datasets import load_dataset
from haystack import Document

dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

from haystack.components.embedders import SentenceTransformersDocumentEmbedder
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()

docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])

from haystack.components.embedders import SentenceTransformersTextEmbedder
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
retriever = InMemoryEmbeddingRetriever(document_store)

from typing import List
from haystack import Pipeline, component
from transformers import pipeline

# Inicializando o modelo de question-answering
model_name = 'pierreguillou/bert-large-cased-squad-v1.1-portuguese'
qa_pipeline = pipeline("question-answering", model=model_name)

@component
class QAPipelineComponent:
    """
    A component for question answering using a Hugging Face pipeline
    """
    def __init__(self, qa_pipeline):
        self.qa_pipeline = qa_pipeline

    @component.output_types(answers=List[dict])
    def run(self, query: str, documents: List[Document]):
        context = ' '.join([doc.content for doc in documents])
        print(f"Context: {context}")
        result = self.qa_pipeline(question=query, context=context)
        print(f"Answer: {result['answer']}")
        return {"answers": [{"answer": result["answer"], "score": result["score"]}]}

# Criando o pipeline de texto
basic_rag_pipeline = Pipeline()
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component(name="qa_component", instance=QAPipelineComponent(qa_pipeline))

# Conectando os componentes entre si
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "qa_component.documents")

question = "What does Rhodes Statue look like?"
response = basic_rag_pipeline.run({
    "text_embedder": {"text": question},
    "qa_component": {"query": question}
})

# Acessando e exibindo a primeira resposta e sua pontuação
first_answer = response["qa_component"]["answers"][0]
answer = first_answer["answer"]
score = first_answer["score"]
print(f"Answer: {answer}")
print(f"Score: {score}")

