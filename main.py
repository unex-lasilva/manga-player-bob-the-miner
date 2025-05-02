import pandas as pd
import numpy as np

# 1. Carregando os dados
ratings = pd.read_csv('data_set/ratings_small.csv')
metadata = pd.read_csv('data_set/movies_metadata.csv', low_memory=False)

# 2. Pré-processamento
ratings = ratings[['userId', 'movieId', 'rating']]
metadata = metadata[['id', 'title', 'genres']]
metadata['id'] = pd.to_numeric(metadata['id'], errors='coerce')
metadata = metadata.rename(columns={'id': 'movieId'})
metadata = metadata.dropna(subset=['movieId'])
metadata['movieId'] = metadata['movieId'].astype(int)

# 3. Unificando os dados
df = pd.merge(ratings, metadata, on='movieId')
df = df[df['rating'] >= 3.0]  # Apenas filmes curtidos

# 4. Amostragem de usuários (evita estouro de memória)


# 5. Criando o mapa {userId: [filmes curtidos]}
user_movies_map = df.groupby('userId')['title'].apply(list).to_dict()



# Dataset
movie_dataset = user_movies_map


Dataset = list(movie_dataset.values())

min_support = 0.3
min_confidence = 0.4

# Função de combinações
def generate_combinations(items, k):
    results = []

    def backtrack(start, path):
        if len(path) == k:
            results.append(path[:])
            return
        for i in range(start, len(items)):
            path.append(items[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return results

# Suporte
def count_support(itemset, dataset):
    count = sum(1 for transaction in dataset if all(item in transaction for item in itemset))
    return count / len(dataset)

# Confiança
def confidence(X, Y, dataset):
    count_X = 0
    count_XY = 0
    for transaction in dataset:
        if all(x in transaction for x in X):
            count_X += 1
            if all(y in transaction for y in Y):
                count_XY += 1
    return count_XY / count_X if count_X > 0 else 0

# Lift
def lift(X, Y, dataset):
    support_X = count_support(X, dataset)
    support_Y = count_support(Y, dataset)
    conf = confidence(X, Y, dataset)
    return conf / support_Y if support_Y > 0 else 0


# Apriori
def get_frequent_itemsets_by_level(dataset, min_support):
    # Nível 1 - conta todos os itens individualmente
    item_counts = {}
    for transaction in dataset:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1
    total_transactions = len(dataset)
    level_1 = [(tuple([item]), count / total_transactions) for item, count in item_counts.items() if (count / total_transactions) >= min_support]

    levels = {}
    if not level_1:
        return levels
    levels[1] = level_1

    k = 2
    current_itemsets = [list(t[0]) for t in level_1]

    while current_itemsets:
        # Gerar candidatos
        candidates = []
        items = sorted(set(item for subset in current_itemsets for item in subset))
        candidates = generate_combinations(items, k)

        # Filtrar por suporte
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

# Regras de associação
def generate_association_rules(frequent_itemsets, dataset, min_confidence):
    rules = []
    for itemset, support in frequent_itemsets:
        if len(itemset) < 2:
            continue
        items = list(itemset)
        n = len(items)
        for i in range(1, n):
            subsets = generate_combinations(items, i)
            for antecedent in subsets:
                consequent = [item for item in items if item not in antecedent]
                if not consequent:
                    continue
                conf = confidence(antecedent, consequent, dataset)
                if conf >= min_confidence:
                    rules.append((
                        tuple(antecedent),
                        tuple(consequent),
                        conf,
                        lift(antecedent, consequent, dataset)
                    ))
    return rules

# Executar Apriori
frequent_levels = get_frequent_itemsets_by_level(Dataset, min_support)

# Tabela de itemsets
all_itemsets = []
for level, itemsets in frequent_levels.items():
    for itemset, support in itemsets:
        all_itemsets.append({
            "Nível": level,
            "Itemset": itemset,
            "Suporte": round(support, 2)
        })
    

df_itemsets = pd.DataFrame(all_itemsets)
print("Itemsets Frequentes:")
print(df_itemsets)

# Tabela de regras
all_frequent_itemsets = [item for sublist in frequent_levels.values() for item in sublist]
rules = generate_association_rules(all_frequent_itemsets, Dataset, min_confidence)

rules_data = []
for antecedent, consequent, conf, lift_value in rules:
    rules_data.append({
        "Antecedente": antecedent,
        "Consequente": consequent,
        "Confiança": round(conf, 2),
        "Lift": round(lift_value, 2)
    })

df_rules = pd.DataFrame(rules_data)
print("\nRegras de Associação:")
print(df_rules)


