# Nazwy modeli:

- vllm/deepSeek-v2-lite-chat
- vllm/qwen2.5-32b-instruct
- vllm/llama-3-70b-instruct-awq
- vllm/qwen2-72b-instruct-awq
- vllm/mistral-small-24b-instruct-2501

## Konfiguracje modeli:
Do konfiguracji nowego/istniejącego modelu musimy wprowadzić zmiany w 3 plikach
- model_deployments.yaml
- model_metadata.yaml
- tokenizer_configs.yaml

1) tokenizer_configs.yaml
```yaml
- name: <nazwa_modelu>
  tokenizer_spec:
    class_name: "helm.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer"
  end_of_text_token: <koniec>
  prefix_token: <początek>
```
2) model_metadata.yaml
```yaml
  - name: <nazwa_modelu>
    display_name: <nazwa_do_wyświetlania>
    description: <opis modelu>
    creator_organization_name: vllm
    access: open
    num_parameters: <liczba_parametrów>
    release_date: 2023-02-13
    tags: [TEXT_MODEL_TAG, PARTIAL_FUNCTIONALITY_TEXT_MODEL_TAG]
```
3) model_deployments.yaml
```yaml
  - name: <nazwa>
    model_name: <nazwa_modelu>
    tokenizer_name: <nazwa_tokenizera>
    max_sequence_length: 2048
    client_spec:
      class_name: "helm.clients.vllm_client.VLLMClient"
      args:
        base_url:  http://127.0.0.1:8000/v1/

```


# Uruchamianie benchmarku:
```shell
helm-run --run-entries <scenariusz:parametr=wartość>,model=<nazwa_modelu> \
         --suite my-suite \
         --max-eval-instances <liczba>
```

# Scenariusze
1) MeQSum (plik me_q_sum_scenario.py) - w opisie jest wzmianka o ROUGE-1
2) TruthfulQA (plik truthful_qa_scenario.py) - parametr task, wartość "mc_single" 
3) Summarization (plik summarization_scenario.py) - dostępne zbiory Xsum i cnn-dm 
4) Legal summarization (plik legal_summarization_scenario.py) - parametr dataset_name, parametry: BillSum, MultiLexSum, EurLexSum 
5) Disinformation (plik disinformation_scenario.py) - parametr capability: reiteration, wedging 
6) BBQ (plik bbq_scenario.py) - parametr subject, wartości:
   - "Age",
   - "Disability_status",
   - "Gender_identity",
   - "Nationality",
   - "Physical_appearance",
   - "Race_ethnicity",
   - "Race_x_SES",  # extra intersectional category as mentioned in section 3.2
   - "Race_x_gender",  # extra intersectional category as mentioned in section 3.2
   - "Religion",
   - "SES",
   - "Sexual_orientation" 

7) MMLU (plik mmlu_scenario.py) - chyba tu jest lista opcji https://crfm.stanford.edu/helm/mmlu/latest/#/scenarios