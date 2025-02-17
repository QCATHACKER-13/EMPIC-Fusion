# Define the dimensions of the 2D array
rows = 3
cols = 3

# Create a 2D array filled with zeros
a1_2d = [[0 for _ in range(cols)] for _ in range(rows)]
a2_2d = [[0 for _ in range(cols)] for _ in range(rows)]


# Using a for loop to store values in the 2D array
value1 = 0
value2 = 10

for i in range(rows):
    for j in range(cols):
        value1 += 1
        value2 -= 1
        a1_2d[i][j] = value1
        a2_2d[i][j] = value2
        a1_2d[i][j] += a2_2d[i][j]

#for i in range(rows):
    #for j in range(cols):
        #a1_2d[i][j] += value1

#for i in range(rows):
    #for j in range(cols):
        #a2_2d[i][j] -= value2

# Printing the 2D array
for row1 in a1_2d:
    print(row1)
for row2 in a2_2d:
    print(row2)

print(a2_2d[0][:])
