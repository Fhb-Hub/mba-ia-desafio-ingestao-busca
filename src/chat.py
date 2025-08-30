from search import search_prompt

SEPARATOR = "-" * 50


def main():
    print("Bem-vindo ao Chat RAG CLI!")
    print("Digite sua pergunta ou 'sair' para encerrar o chat.")
    print(SEPARATOR)
    handle_user_interaction()


def handle_user_interaction():
    while True:
        try:
            user_question = input("Pergunta: ")
            if user_question.lower() == "sair":
                break

            generated_response = search_prompt(user_question)
            print("\nResposta gerada:", generated_response)
            print(SEPARATOR)

        except KeyboardInterrupt:
            print("\nChat encerrado pelo usuário.")
            break
        except Exception as e:
            print(f"\nOcorreu um erro: {e}")
            break


if __name__ == "__main__":
    main()
