## Binary Search

Original code:
```
def binary_search(nums, target):
		left, right = 0, len(nums)-1
		while left <= right:
				mid = (left+right)//2
				if nums[mid] == target:
						return mid
				elif nums[mid] < target:
						left = mid + 1
				else:
						right = mid - 1
						
		return -1
```

The loop can get “stuck” — e.g., if `left` and `right` stop changing, causing an infinite loop. Each iteration **strictly shrinks the search space**: That means there’s **no case** where `left` and `right` remain the same between iterations — the interval always shrinks until the search terminates.

- **Time:** O(log n)
- **Space:** O(1)

### Variations in LeetCode
528. Random Pick with Weight: the key is to realize that the binary search is helping the search efficiency when we have a random index to search from the prefix sum space. 

875. Koko Eating Bananas:  the key part is to realize that we have to iterate over a range of K values, which will not make the O() more complex, then the problem is how can we iterate the K faster, which led to the solution of binary search. 
