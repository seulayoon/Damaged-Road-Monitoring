from bs4 import BeautifulSoup
import cv2
import glob

# 파싱
dirPath = "/home/piai/A2 PROJECT/dataset_zip/Surface_1/*"
dirlist = glob.glob(dirPath)


for surface_dir in dirlist:
    xmlpath = glob.glob(surface_dir + '/*.xml')
    xmlpath = ",".join(xmlpath)

    fp = open(xmlpath, "r")
    soup = BeautifulSoup(fp, "lxml-xml")

    imglist = []

    images = soup.select('image')
    for image in images:
        tags = image.select('polygon > attribute')
        for tag in tags:
            if tag.text == 'damaged':
                imglist.append(image.get('name'))

    # 중복 제거
    imgset = set(imglist)
    dmglist = list(imgset)

    for img in dmglist:
        
        # original data
        # read
        imgPath = surface_dir + '/' + img
        original = cv2.imread(imgPath)
        # save
        savePath = '/home/piai/Desktop/Sur1_damaged/' + img
        cv2.imwrite(savePath, original)

        # mask png
        # read
        img = img.replace('jpg', 'png')
        imgPath = surface_dir + '/MASK/' + img
        original = cv2.imread(imgPath)
        # save
        savePath = '/home/piai/Desktop/Sur1_damaged/MASK/' + img
        cv2.imwrite(savePath, original)


fp.close()
