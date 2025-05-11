import pandas as pd
import numpy as np

# 1. Carregando os dados
ratings = pd.read_csv('data_set/ratings_small.csv')
metadata = pd.read_csv('data_set/movies_metadata.csv', low_memory=False)

# 2. Pré-processamento
ratings = ratings[['userId', 'movieId', 'rating']]
metadata = metadata[['id', 'title', 'genres']]
metadata['id'] = pd.to_numeric(metadata['id'], errors='coerce')
metadata = metadata.dropna(subset=['id'])
metadata['id'] = metadata['id'].astype(int)
metadata = metadata.rename(columns={'id': 'movieId'})

# 3. Unificando os dados
df = pd.merge(ratings, metadata, on='movieId')
df = df[df['rating'] >= 3.0]  # Apenas filmes curtidos

# 4. Criando o mapa {userId: [filmes curtidos]}
user_movies_map = df.groupby('userId')['title'].apply(list).to_dict()
Dataset = list(user_movies_map.values())



# Parâmetros
min_support = 0.1
min_confidence = 0.5
max_level = 4  # Limite de níveis para o Apriori
IdUser = int(input("Informe o Usuario: "))  # ID do usuário para exemplo de recomendação



# Combinações
def generate_combinations(items, k):
    results = []
    def backtrack(start, path):
        if len(path) == k:
            results.append(tuple(path))
            return
        for i in range(start, len(items)):
            path.append(items[i])
            backtrack(i + 1, path)
            path.pop()
    backtrack(0, [])
    return results

# Suporte
def count_support(itemset, dataset):
    return sum(1 for transaction in dataset if all(item in transaction for item in itemset)) / len(dataset)

# Confiança
def confidence(X, Y, dataset):
    count_X = sum(1 for t in dataset if all(x in t for x in X))
    count_XY = sum(1 for t in dataset if all(x in t for x in X + Y))
    return count_XY / count_X if count_X > 0 else 0

# Lift
def lift(X, Y, dataset):
    support_X = count_support(X, dataset)
    support_Y = count_support(Y, dataset)
    conf = confidence(X, Y, dataset)
    return conf / support_Y if support_Y > 0 else 0

# Apriori (com limite até L4)
def get_frequent_itemsets_by_level(dataset, min_support, max_level):
    item_counts = {}
    for transaction in dataset:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1

    total_transactions = len(dataset)

    # L1: todos os itens, sem filtrar por suporte
    level_1 = [(tuple([item]), count / total_transactions) for item, count in item_counts.items()]
    levels = {}
    if not level_1:
        return levels
    levels[1] = level_1

    # Filtra apenas os itens com suporte >= min_support para uso nas combinações
    frequent_l1_items = [item for item, count in item_counts.items() if (count / total_transactions) >= min_support]
    level_1 = [(tuple([item]), count / total_transactions) for item, count in item_counts.items()]  # mantém todos no L1
    levels[1] = level_1

    k = 2
    current_itemsets = [[item] for item in frequent_l1_items]

    while current_itemsets and k <= max_level:
        unique_items = sorted(set(item for subset in current_itemsets for item in subset))
        candidates = generate_combinations(unique_items, k)
        level_frequent = []
        for itemset in candidates:
            support = count_support(itemset, dataset)
            if support >= min_support:
                level_frequent.append((tuple(itemset), support))
        if not level_frequent:
            break
        levels[k] = level_frequent
        current_itemsets = [list(t[0]) for t in level_frequent]
        k += 1

    return levels

# Geração de regras + dicionário para acesso rápido
def generate_association_rules(frequent_itemsets, dataset, min_confidence):
    rules = []
    rules_map = {}
    for itemset, support in frequent_itemsets:
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for i in range(1, len(items)):
            for antecedent in generate_combinations(items, i):
                antecedent = tuple(antecedent)
                consequent = tuple(item for item in items if item not in antecedent)
                if not consequent:
                    continue
                conf = confidence(list(antecedent), list(consequent), dataset)
                if conf >= min_confidence:
                    lft = lift(list(antecedent), list(consequent), dataset)
                    rule = (antecedent, consequent, conf, lft)
                    rules.append(rule)
                    if antecedent not in rules_map:
                        rules_map[antecedent] = []
                    rules_map[antecedent].append((consequent, conf, lft))
    return rules, rules_map

# Recomendação por histórico
def recomendar_por_historico(usuario_filmes, rules_map, dataset):
    recomendacoes = {}
    for antecedente in rules_map:
        if all(f in usuario_filmes for f in antecedente):
            for consequente, conf, _ in rules_map[antecedente]:
                for filme in consequente:
                    if filme not in usuario_filmes and filme not in recomendacoes:
                        suporte = count_support([filme], dataset)
                        recomendacoes[filme] = (conf, suporte)
    recomendacoes_ordenadas = sorted(recomendacoes.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)
    return [filme for filme, _ in recomendacoes_ordenadas[:4]]

# Recomendação por último filme curtido
def recomendar_por_ultimo_filme(usuario_filmes, regras, dataset, min_conf_threshold=0.01, max_tentativas=5):
    recomendacoes = {}
    tentativas = 0
    for filme in reversed(usuario_filmes):
        tentativas += 1
        for antecedente, consequente_list in regras.items():
            for consequente, conf, lift_val in consequente_list:
                if conf < min_conf_threshold:
                    continue
                if filme in antecedente:
                    for f in consequente:
                        if f not in usuario_filmes and f not in recomendacoes:
                            suporte = count_support([f], dataset)
                            recomendacoes[f] = (conf, suporte)
                elif filme in consequente:
                    for f in antecedente:
                        if f not in usuario_filmes and f not in recomendacoes:
                            suporte = count_support([f], dataset)
                            recomendacoes[f] = (conf, suporte)
        if recomendacoes or tentativas >= max_tentativas:
            break
    recomendacoes_ordenadas = sorted(recomendacoes.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)
    return [filme for filme, _ in recomendacoes_ordenadas[:4]]

# Recomendação baseada no L1
def recomendar_com_L1(usuario_filmes, frequent_levels, top_n=4):
    l1 = frequent_levels.get(1, [])
    l1_sorted = sorted(l1, key=lambda x: x[1], reverse=True)
    recomendacoes = [item[0][0] for item in l1_sorted if item[0][0] not in usuario_filmes]
    return recomendacoes[:top_n]

# Executar Apriori até L1
frequent_levels = get_frequent_itemsets_by_level(Dataset, min_support, max_level)
all_frequent_itemsets = [item for sublist in frequent_levels.values() for item in sublist]

# Tabela de itemsets
df_itemsets = pd.DataFrame([
    {"Nível": k, "Itemset": itemset, "Suporte": round(support, 2)}
    for k, itemsets in frequent_levels.items() for itemset, support in itemsets
])
print(" Itemsets Frequentes:")
print(df_itemsets)

# Geração de regras com map
rules, rules_map = generate_association_rules(all_frequent_itemsets, Dataset, min_confidence)
df_rules = pd.DataFrame([
    {"Antecedente": a, "Consequente": c, "Confiança": round(conf, 2), "Lift": round(lf, 2)}
    for a, c, conf, lf in rules
])
print("\n Regras de Associação:")
print(df_rules)

# Exemplo de recomendações
exemplo_user_id = list(user_movies_map.keys())[IdUser]
filmes_usuario = user_movies_map[exemplo_user_id]

print(f"\n Filmes curtidos pelo usuário {exemplo_user_id}:")
print(filmes_usuario)

# Histórico
recomendacoes_hist = recomendar_por_historico(filmes_usuario, rules_map, Dataset)
if recomendacoes_hist:
    print("\n Top 4 Recomendações baseadas no histórico completo: ")
    print(recomendacoes_hist)
else:
    # Fallback
    print("\n Nenhuma recomendação por histórico. Mas aqui vão algumas sugestões populares!")
    print(recomendar_com_L1(filmes_usuario, frequent_levels))

# Último curtido
recomendacoes_ult = recomendar_por_ultimo_filme(filmes_usuario, rules_map, Dataset)
if recomendacoes_ult:
    print("\n Top 4 Recomendações baseadas no último filme curtido:")
    print(recomendacoes_ult)
else:
    # Fallback
    print("\n Nenhuma recomendação pelo último filme curtido. Mas aqui vão algumas sugestões populares!")
    print(recomendar_com_L1(filmes_usuario, frequent_levels))

print(f"\nTotal de filmes curtidos pelo usuário: {len(filmes_usuario)}")

