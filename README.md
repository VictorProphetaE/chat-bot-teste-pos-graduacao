# ChatBot

Este é um ChatBot simples que utiliza técnicas de processamento de linguagem natural (NLP) para responder a perguntas e manter conversas básicas. O ChatBot é treinado com base em arquivos JSON contendo intenções e respostas.

## Pré-requisitos

Certifique-se de ter as seguintes bibliotecas instaladas:

- nltk
- tflearn
- tensorflow
- numpy

Você pode instalar as dependências usando o seguinte comando:

	pip install nltk tflearn tensorflow numpy
	

## Como usar

Siga as etapas abaixo para usar o ChatBot:

1. Execute a função `merge_json_files` para mesclar os arquivos JSON contendo as intenções em um único arquivo chamado "merged.json". Certifique-se de ter os arquivos JSON corretamente estruturados e localizados no diretório atual.

2. Execute a função `preprocess_data` para processar os dados das intenções e preparar os dados de treinamento. Os dados processados serão salvos em um arquivo chamado "data.pickle".

3. Execute a função `train_model` para treinar o modelo do ChatBot usando os dados de treinamento processados. O modelo treinado será salvo em um arquivo chamado "model.tflearn".

4. Opcionalmente, você pode executar a função `add_new_word` para adicionar novas palavras e respostas às intenções existentes. As palavras e respostas adicionadas serão salvas em um arquivo chamado "storedata.json".

5. Finalmente, execute a função `chat` para iniciar uma conversa com o ChatBot. O ChatBot responderá às suas perguntas com base nas intenções treinadas.

Certifique-se de ter os arquivos "merged.json", "data.pickle" e "model.tflearn" no diretório atual antes de executar a função `chat`. Caso contrário, você receberá uma mensagem de aviso indicando que nenhum arquivo de dados foi encontrado.

## Notas

- Os arquivos JSON contendo as intenções devem ter a seguinte estrutura:

```json
{
  "intents": [
    {
      "tag": "saudacao",
      "patterns": ["Oi", "Olá", "Bom dia"],
      "responses": ["Olá, como posso ajudar?", "Oi, o que você precisa?", "Bom dia! Em que posso ser útil?"]
    },
    {
      "tag": "despedida",
      "patterns": ["Tchau", "Até logo", "Adeus"],
      "responses": ["Até logo! Volte sempre.", "Tchau! Tenha um bom dia.", "Adeus, até a próxima."]
    },
    ...
  ]
}

O arquivo "merged.json" será criado durante a execução da função merge_json_files. Se você adicionar novas intenções posteriormente, lembre-se de executar novamente essa função para atualizar o arquivo.

Se você adicionar novas palavras e respostas usando a função add_new_word, certifique-se de executar novamente as etapas 2 e 3 (funções preprocess_data e train_model) para processar os dados atualizados e re-treinar o modelo.

O modelo treinado será salvo em "model.tflearn". Se você deseja treinar o modelo