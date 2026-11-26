package ht.mbds.Tarrouzi;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import java.util.List;
import java.util.Scanner;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
public class RagNaif {
    public static void main(String[] args) {
        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        DocumentSplitter splitter = DocumentSplitters.recursive(800, 100);
        List<TextSegment> segments = splitter.split(document);

        System.out.println("Segments extraits : " + segments.size());
        EmbeddingModel embeddingModel = new BgeSmallEnV15EmbeddingModel();

        Response<List<Embedding>> response = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = response.content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        System.out.println("Embeddings créés : " + embeddings.size());


        String apiKey = System.getenv("GEMINI_API_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("Définis GEMINI_API_KEY dans la configuration Run.");
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash")
                .build();

        ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .contentRetriever(retriever)
                .build();

//        String question = "Quelle est la signification de 'RAG' et à quoi ça sert ?";
//        String reponse = assistant.chat(question);
//        System.out.println(reponse);


        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.print("Question (ou 'quitter') > ");

                if (!scanner.hasNextLine()) {
                    break;
                }

                String question = scanner.nextLine().strip();
                if (question.equalsIgnoreCase("quitter")) {
                    break;
                }
                if (question.isBlank()) {
                    continue;
                }

                String reponse = assistant.chat(question);
                System.out.println("Gemini > " + reponse);
            }
        }
    }
}