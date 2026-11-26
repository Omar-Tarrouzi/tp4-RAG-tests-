package ht.mbds.Tarrouzi;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class test3 {

    public static void main(String[] args) {
        String apiKey = System.getenv("GEMINI_API_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException(
                    "ajouter clé");
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash")
                .logRequestsAndResponses(true)
                .build();

        EmbeddingModel embeddingModel = new BgeSmallEnV15EmbeddingModel();
        ContentRetriever ragRetriever = creerRetriever("rag.pdf", embeddingModel);


        PromptTemplate template = PromptTemplate.from(
                "Est-ce que la requête « {{question}} » porte sur l'IA ? "
                        + "Réponds uniquement par oui, non ou peut-être.");

        QueryRouter router = new QueryRouter() {
            @Override
            public Collection<ContentRetriever> route(Query query) {
                String prompt = template.apply(Map.of("question", query.text())).text();
                String decision = model.chat(prompt).strip();

                if (decision.equalsIgnoreCase("non")) {
                    return List.of();
                }
                return List.of(ragRetriever);
            }
        };



        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(augmentor)
                .build();

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

                System.out.println("Gemini > " + assistant.chat(question));
            }
        }
    }

    private static ContentRetriever creerRetriever(
            String nomPdf, EmbeddingModel embeddingModel) {
        Document document = ClassPathDocumentLoader.loadDocument(
                nomPdf, new ApacheTikaDocumentParser());

        List<TextSegment> segments = DocumentSplitters.recursive(800, 100)
                .split(document);

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddingModel.embedAll(segments).content(), segments);

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }
}