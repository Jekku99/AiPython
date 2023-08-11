def bubble_sort(luvut):
	n = len(luvut)
	for i in range(n-1):
			for j in range(n-i-1):
				if luvut[j] > luvut[j+1]:
					luvut[j], luvut[j+1] = luvut[j+1], luvut[j]
	return luvut
