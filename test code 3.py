# Open a file in write mode
for j in range(5):
    with open('output.txt', 'w') as file:
    # Your loop starts here
    for i in range(10):  # Replace 10 with the desired number of iterations
        # Your data to be saved in each iteration
        data_to_save = f'Iteration {i+1}: This is some data.\n'
        
        # Write the data to the file
        file.write(data_to_save)
        
        # Optionally, print the data for verification
        print(data_to_save)
        
# The file will be automatically closed when the 'with' block is exited



    
