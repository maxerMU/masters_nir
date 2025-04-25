EPOCHS = 42

# with open("yolo_error.dat", "w") as f:
#     for x in range(1, EPOCHS + 1):
#         y = 1 / x + 0.1
#         f.write(f"{x} {y}\n")

with open("yolo_test.dat", "w") as f:
    for x in range(1, EPOCHS + 1):
        y = 1 / (1 + pow(2.7, -(x / 42)))
        f.write(f"{x} {y}\n")