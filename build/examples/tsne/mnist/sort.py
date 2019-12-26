#!/usr/bin/env python

def main():
    with open('images.txt', encoding='utf-8') as inf:
        images = inf.readlines()
    
    with open('labels.txt', encoding='utf-8') as inf:
        labels = [int(line.strip()) for line in inf.readlines()]

    with open('images.sorted.txt', 'w', encoding='utf-8') as outf1:
        with open('labels.sorted.txt', 'w', encoding='utf-8') as outf2:
            for image, label in sorted(zip(images, labels), key=lambda x:x[1]):
                print(image.strip(), file=outf1)
                print(label, file=outf2)

if __name__ == '__main__':
    main()
