# AI Image Finder — V4

Projekt realizujący pipeline:

**zdjęcie → opis AI → embedding → Qdrant → wyszukiwanie semantyczne**

## Funkcje V4

1. Użytkownik może przesłać jedno lub wiele zdjęć.
2. Zdjęcia są zapisywane lokalnie w `data/images/`.
3. Model `gpt-4o` analizuje każde zdjęcie i generuje opis po polsku.
4. Opis jest zamieniany na embedding przez `text-embedding-3-large`.
5. Embedding + metadata są zapisywane w lokalnym, trwałym Qdrant.
6. Użytkownik wpisuje opis, np.:
   - `czerwony samochód na parkingu`
   - `osoba siedząca przy biurku`
   - `budynek obok samochodu`
7. Zapytanie jest również zamieniane na embedding.
8. Qdrant zwraca semantycznie najbardziej podobne zdjęcia.

## Architektura

```text
                    ┌───────────────────┐
                    │  Streamlit UI     │
                    └─────────┬─────────┘
                              │
                upload zdjęcia│
                              ▼
                    ┌───────────────────┐
                    │ data/images/      │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ GPT-4o Vision     │
                    │ opis zdjęcia      │
                    └─────────┬─────────┘
                              │
                           opis tekstowy
                              │
                              ▼
                    ┌───────────────────┐
                    │ text-embedding-   │
                    │ 3-large           │
                    └─────────┬─────────┘
                              │
                         embedding
                              │
                              ▼
                    ┌───────────────────┐
                    │ Qdrant            │
                    │ vector + metadata │
                    └─────────┬─────────┘
                              ▲
                              │
                    embedding zapytania
                              │
                    ┌─────────┴─────────┐
                    │ wyszukiwanie      │
                    │ użytkownika       │
                    └───────────────────┘
```

## Dlaczego lokalny Qdrant?

Na potrzeby projektu nie trzeba uruchamiać osobnego serwera Qdrant.

`QdrantClient(path="data/qdrant")` przechowuje dane lokalnie i dzięki temu po restarcie aplikacji kolekcja pozostaje dostępna.

Później można bez dużej zmiany architektury przejść na Qdrant Cloud / serwer Qdrant.

## Instalacja

### 1. Virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Biblioteki

```bash
pip install -r requirements.txt
```

### 3. Klucz OpenAI

Skopiuj:

```text
.env.example
```

do:

```text
.env
```

i wpisz:

```text
OPENAI_API_KEY=sk-...
```

### 4. Uruchomienie

```bash
streamlit run app.py
```

## Struktura projektu

```text
image_finder_v4/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── image_service.py
│   ├── openai_service.py
│   └── vector_store.py
│
└── data/
    ├── images/
    └── qdrant/
```

## V1 → V4

### V1
Upload i zapis zdjęć.

### V2
Do każdego zdjęcia generowany jest opis przez model vision.

### V3
Opis → embedding → zapis w Qdrant.

### V4
Zapytanie użytkownika → embedding → similarity search → zdjęcia pasujące semantycznie.

W tej wersji wszystkie cztery kroki są połączone w jeden działający pipeline.

## Ważne

`text-embedding-3-large` domyślnie zwraca wektor 3072-wymiarowy, dlatego kolekcja Qdrant jest tworzona z:

```python
VectorParams(
    size=3072,
    distance=Distance.COSINE,
)
```

Jeżeli później zmienisz model embeddingów na model o innym rozmiarze wektora, trzeba zmienić konfigurację kolekcji.

## Co można dodać jako V5

- usuwanie zdjęć,
- ponowne generowanie opisów,
- filtrowanie po dacie / nazwie / tagach,
- paginację,
- hybrid search,
- reranking,
- bezpośrednie embeddingi obrazów zamiast embeddingów opisów,
- Docker + Qdrant server,
- PostgreSQL na metadata,
- testy jednostkowe,
- async/batch processing,
- kolejkę zadań dla dużej liczby zdjęć,
- logowanie kosztów API.
