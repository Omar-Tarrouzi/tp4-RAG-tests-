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
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class Test4 {

    public static void main(String[] args) {
        String apiKey = lireCle("GEMINI_API_KEY");
        String tavilyApiKey = lireCle("TAVILY_API_KEY");

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash")
                .logRequestsAndResponses(true)
                .build();

        EmbeddingModel embeddingModel = new BgeSmallEnV15EmbeddingModel();
        ContentRetriever ragRetriever = creerRetriever("rag.pdf", embeddingModel);

        WebSearchEngine searchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyApiKey)
                .build();

        ContentRetriever webRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(searchEngine)
                .build();

        QueryRouter router = creerRouteur(args, model, ragRetriever, webRetriever);

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

    private static QueryRouter creerRouteur(
            String[] args, ChatModel model, ContentRetriever ragRetriever,
            ContentRetriever webRetriever) {
        // Version initiale : new DefaultQueryRouter(ragRetriever, webRetriever).
        if (List.of(args).contains("--sans-routage")) {
            return new DefaultQueryRouter(ragRetriever, webRetriever);
        }

        return new LanguageModelQueryRouter(model, Map.of(
                ragRetriever, "Support de cours sur le RAG et le fine-tuning. "
                        + "À choisir pour les questions sur ces sujets ou sur le support de cours.",
                webRetriever, "Recherche Web pour les faits récents et les sujets hors du support. "
                        + "À choisir seul pour les questions d'actualité sans rapport avec le cours."
        ));
    }

    private static String lireCle(String variable) {
        String valeur = System.getenv(variable);
        if (valeur == null || valeur.isBlank()) {
            throw new IllegalStateException("Définis la variable d'environnement " + variable + ".");
        }
        return valeur;
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
