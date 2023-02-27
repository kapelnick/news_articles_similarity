import os

def print_files(directory):
    i = 1
    for root, dirs, files in os.walk(directory):
        print(f"Directory: {root}")
        for file in files:
            #print(f"\t{file} " + str(i))
            i = i + 1
        print(i)
        i = 0

# Example usage:
print_files("C:/Users/kapel/Desktop/news_1")
#print_files("path/to/directory")