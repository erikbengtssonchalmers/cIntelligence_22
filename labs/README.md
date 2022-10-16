# Lab 1: Computational Intelligence 22

Lab solution was performed with the following discussion partners:

- Karl Wennerström,
- Sidharrth Nagappan,
- Angelica Ferlin

However, the implementation is my own with inspiration and structure from the instructions given in the task description by Giovanni.

The implementation is a basic algorithm that sorts the input based on the most unique numbers. By unique numbers I mean numbers that hasn't been
discovered yet. It removes the first entry from the sorted list and adds it
to solution set (and path) if there's a unique entry. In every iteration, the list is sorted 
which is time-consuming.

### Results
 - Success: N=5, weight=6 
 - Success: N=10, weight=13 
 - Success: N=20, weight=32 
 - Success: N=100, weight=191 
 - Success: N=500, weight=1375 
 - Success: N=1000, weight=3087