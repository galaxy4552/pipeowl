from engine import PipeOwlEngine, PipeOwlConfig

engine = PipeOwlEngine(PipeOwlConfig())

while True:
    query = input("請輸入句子： ")

    out = engine.pipeowl(query, top_k=5)

    print("\nTop-K Tokens:")
    for text, score in out["retrieved"]:
        print(f"{score:.3f} | {text}")

    print("\nDecoded:")
    print(out["decoded"])
    print()
