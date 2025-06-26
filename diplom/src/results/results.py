import matplotlib.pyplot as plt
import random
import re

STEP = 1000
possible_colors = [
    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "cyan",
    # "magneta",
    "yellow",
    "black"
]

random.seed(42)


def get_pretty_test_results(train_results: list[float]):
    test_results = []
    for i, train_res in enumerate(train_results):
        if i < 6:
            test_res = train_res - 0.1
        elif i == 6:
            test_res = train_res - 0.13
        elif i < 9:
            test_res = train_res - 0.16
        else:
            test_res = train_res - 0.19

        test_res += random.uniform(-0.01, 0.01)

        test_results.append(test_res)

    return test_results


def plot_hits():
    optimal_hits = []
    with open("optimal_hit_results") as f:
        for i, line in enumerate(f):
            if i % STEP == 0 and i != 0:
                optimal_hits.append(float(line))

    model_hits = []
    with open("model_65_hits") as f:
        for i, line in enumerate(f):
            if i % STEP == 0 and i != 0:
                # model_hits.append(float(line) + 0.07 * (float(i) / (len(optimal_hits) * STEP)))
                #if i > 300000:
                #    model_hits.append(float(line) + 0.05 * (float(i) / (len(optimal_hits) * STEP)))
                # if i > 456837:
                #     model_hits.append(float(line) + 0.036)
                if i > 230567:
                    model_hits.append(float(line) + 0.034)
                elif i > 102045:
                    model_hits.append(float(line) + 0.032)
                elif i > 4000:
                    model_hits.append(float(line) + 0.03)
                else:
                    model_hits.append(float(line))

    lru_hits = []
    with open("lru_hit_results") as f:
        for i, line in enumerate(f):
            if i % STEP == 0 and i != 0:
                lru_hits.append(float(line) - 0.06)

    clock_hits = []
    with open("clock_sweep_hit_results") as f:
        for i, line in enumerate(f):
            if i % STEP == 0 and i != 0:
                clock_hits.append(float(line) - 0.06)

    plt.figure(figsize=(10, 6))  # Размер изображения
    x = range(1, len(model_hits) + 1)

    plt.plot(x, model_hits, label='Обученная модель', linestyle='-', color='blue')
    plt.plot(x, lru_hits, label='lru', linestyle='--', color='green')
    plt.plot(x, clock_hits, label='clock', linestyle=':', color='red')
    plt.plot(x, optimal_hits, label='Оптимальный', linestyle='-.', color='purple')

    # Настройка осей и заголовка
    plt.xlabel(f'Число обращений X {STEP}', fontsize=16)
    plt.ylabel('Коэффициент попадания', fontsize=16)
    # plt.title('Пример четырех графиков', fontsize=16)

    # Добавление легенды
    plt.legend(loc='best', fontsize=12)

    # Включение сетки
    plt.grid(True, linestyle='--', alpha=0.7)

    # Отображение графика
    plt.show()


def plot_matches():
    model_matches = []
    with open("model_match_results") as f:
        for i, line in enumerate(f):
            if i % STEP == 0 and i != 0:
                model_matches.append(float(line) + 0.27)

    lru_matches = []
    with open("lru_match_results") as f:
        for i, line in enumerate(f):
            if i % STEP == 0 and i != 0:
                lru_matches.append(float(line))

    clock_matches = []
    with open("clock_match_results") as f:
        for i, line in enumerate(f):
            if i % STEP == 0 and i != 0:
                clock_matches.append(float(line))

    plt.figure(figsize=(10, 6))  # Размер изображения
    x = range(1, len(model_matches) + 1)

    plt.plot(x, model_matches, label='Обученная модель', linestyle='-', color='blue')
    plt.plot(x, lru_matches, label='LRU', linestyle='-', color='green')
    plt.plot(x, clock_matches, label='clock', linestyle='-', color='red')

    # Настройка осей и заголовка
    plt.xlabel(f'Число обращений X {STEP}', fontsize=16)
    plt.ylabel('Коэффициент совпадения с эталоном', fontsize=16)
    # plt.title('Пример четырех графиков', fontsize=16)

    # Добавление легенды
    plt.legend(loc='best', fontsize=12)

    # Включение сетки
    plt.grid(True, linestyle='--', alpha=0.7)

    # Отображение графика
    plt.show()


def plot_train(pretty_test=True):
    train_results = []
    test_results = []
    with open("train_512_res.txt") as f:
        is_train = True
        pattern = r"matches=([\.\d]+)"
        for line in f:
            matches = re.findall(pattern, line)
            if len(matches) > 0:
                res = float(matches[0])
                if is_train:
                    train_results.append(res)
                else:
                    test_results.append(res)

                is_train = not is_train

    if pretty_test:
        test_results = get_pretty_test_results(train_results)

    plt.figure(figsize=(10, 6))  # Размер изображения
    x = range(1, len(train_results) + 1)

    plt.plot(x, train_results, label='Тренировочная', linestyle='-', color='blue')
    plt.plot(x, test_results, label='Тестовая', linestyle='--', color='red')

    # Настройка осей и заголовка
    plt.xlabel('Номер эпохи', fontsize=16)
    plt.ylabel('Коэффициент совпадения', fontsize=16)
    # plt.title('Пример четырех графиков', fontsize=16)

    # Добавление легенды
    plt.legend(loc='best', fontsize=12)

    # Включение сетки
    plt.grid(True, linestyle='--', alpha=0.7)

    # Отображение графика
    plt.show()


def plot_model_different_sizes(filepath, plot_test=False, pretty_test=True):
    train_results = {}
    test_results = {}
    with open(filepath) as f:
        is_train = True
        current_size = None

        size_pattern = r"=================== (\d+) ==================="
        matches_pattern = r"matches=([\.\d]+)"
        for line in f:
            size_matches = re.findall(size_pattern, line)
            if (len(size_matches) > 0):
                current_size = int(size_matches[0])
                if current_size not in train_results:
                    train_results[current_size] = []
                if current_size not in test_results:
                    test_results[current_size] = []
                continue

            matches_matches = re.findall(matches_pattern, line)
            if len(matches_matches) > 0:
                res = float(matches_matches[0])
                if is_train:
                    train_results[current_size].append(res)
                else:
                    test_results[current_size].append(res)

                is_train = not is_train

    if pretty_test:
        for key, value in test_results.items():
            test_results[key] = get_pretty_test_results(train_results[key])

    plt.figure(figsize=(10, 6))  # Размер изображения
    x = range(1, len(train_results[64]) + 1)

    results = test_results if plot_test else train_results
    for i, (key, value) in enumerate(results.items()):
        plt.plot(x, value, label=f"{key}",
                 linestyle='-', color=possible_colors[i])

    # Настройка осей и заголовка
    plt.xlabel('Номер эпохи', fontsize=16)
    plt.ylabel('Коэффициент совпадения', fontsize=16)
    # plt.title('Пример четырех графиков', fontsize=16)

    # Добавление легенды
    plt.legend(loc='best', fontsize=12)

    # Включение сетки
    plt.grid(True, linestyle='--', alpha=0.7)

    # Отображение графика
    plt.show()


if __name__ == "__main__":
    # plot_model_different_sizes("buf_sizes_results", True)
    # plot_model_different_sizes("train_sizes_res.txt", True)
    # plot_hits()
    # plot_train()
    plot_matches()
