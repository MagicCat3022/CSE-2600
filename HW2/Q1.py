data = '6, 7, 6, 7, 7, 6, 11, 4, 8, 6, 6, 12, 2, 8, 12, 5, 7, 8, 6, 11, 6, 8, 9, 11, 5, 9, 8, 9, 8, 12'
data = data.split(', ')
data = [int(i) for i in data]
sorted_data = sorted(data)
print(sorted_data)

def Q1_a():
    bin1 = sorted_data[:10]
    bin2 = sorted_data[10:20]
    bin3 = sorted_data[20:]
    print(bin1)
    print(bin2)
    print(bin3)
    bin1_mean = sum(bin1) / len(bin1)
    bin2_mean = sum(bin2) / len(bin2)
    bin3_mean = sum(bin3) / len(bin3)
    print("Bin 1:", bin1_mean)
    print("Bin 2:", bin2_mean)
    print("Bin 3:", bin3_mean)
    bin1 = [bin1_mean] * len(bin1)
    bin2 = [bin2_mean] * len(bin2)
    bin3 = [bin3_mean] * len(bin3)
    print("Bin 1:", bin1)
    print("Bin 2:", bin2)
    print("Bin 3:", bin3)
    
def Q1_b():
    range = max(sorted_data) - min(sorted_data)
    print("Range:", range)
    bin_width = 3
    print("Bin Width:", bin_width)
    bin1 = [x for x in sorted_data if x <= min(sorted_data) + bin_width]
    bin2 = [x for x in sorted_data if min(sorted_data) + bin_width < x <= min(sorted_data) + 2 * bin_width]
    bin3 = [x for x in sorted_data if x > min(sorted_data) + 2 * bin_width]
    print("Bin 1:", bin1)
    print("Bin 2:", bin2)
    print("Bin 3:", bin3)
    bin1_length = len(bin1)
    bin2_length = len(bin2)
    bin3_length = len(bin3)
    print("Bin 1 Length:", bin1_length)
    print("Bin 2 Length:", bin2_length)
    print("Bin 3 Length:", bin3_length)
    bin1_median = bin1[bin1_length // 2] if bin1_length % 2 != 0 else (bin1[bin1_length // 2 - 1] + bin1[bin1_length // 2]) / 2
    bin2_median = bin2[bin2_length // 2] if bin2_length % 2 != 0 else (bin2[bin2_length // 2 - 1] + bin2[bin2_length // 2]) / 2
    bin3_median = bin3[bin3_length // 2] if bin3_length % 2 != 0 else (bin3[bin3_length // 2 - 1] + bin3[bin3_length // 2]) / 2
    print("Bin 1 Median:", bin1_median)
    print("Bin 2 Median:", bin2_median)
    print("Bin 3 Median:", bin3_median)
    bin1 = [bin1_median] * len(bin1)
    bin2 = [bin2_median] * len(bin2)
    bin3 = [bin3_median] * len(bin3)
    print("Bin 1:", bin1)
    print("Bin 2:", bin2)
    print("Bin 3:", bin3)   

def Q1_e():
    mean = sum(sorted_data) / len(sorted_data)
    print("Mean:", mean)
    standard_deviation = (sum((x - mean) ** 2 for x in sorted_data) / len(sorted_data)) ** 0.5
    print("Standard Deviation:", standard_deviation)
    