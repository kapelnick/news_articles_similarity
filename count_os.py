import os

directory = r"C:\Users\kapel\Desktop\dataset"

count = 0
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".txt"):
            count += 1

print("Total number of txt files: ", count)