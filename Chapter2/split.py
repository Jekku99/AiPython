def my_split(sentence, erotin):
    sanat = [] 
    sana = ''
    for i in sentence:
        if i not in erotin:
            sana += i
        else:
            sanat.append(sana)
            sana = ''
    sanat.append(sana)
    return sanat

def my_join(sanat, erotin):
    string = ''
    for i in sanat:
        if i != sanat[-1]:
            sana = i + erotin
            string += sana
        else:
            string += i
    
    return string
        
sentence = str(input("Please enter sentence:"))
print(my_join(my_split(sentence,' '),','))
print(my_join(my_split(sentence,' '),'\n'))
