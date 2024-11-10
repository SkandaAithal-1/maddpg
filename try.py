def generate_grid():
    grid = []
    for i in range(50):
        row = []
        for j in range(50):
            # Border walls
            if i == 0 or i == 49 or j == 0 or j == 49:
                row.append(1)
            # Adding internal walls and paths with narrow corridors
            elif (i % 7 == 2 or j % 7 == 2) and (i % 7 != 0 and j % 7 != 0):
                row.append(1)
            elif (i % 7 == 4 and j % 5 == 3) or (j % 5 == 4 and i % 7 == 3):
                row.append(1)
            else:
                row.append(0)
        grid.append(row)
    return grid

# Display the generated grid
for row in generate_grid():
    print(row)

