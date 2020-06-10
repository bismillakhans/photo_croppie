import dlib
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

class ImageEdit:
    tot_noi = 10  # no of images
    photo_size = [3.5, 4.5]  # in cm default 3.5x4.5 cm
    paper_size = [6, 4]  # in inches default 6x4 inches
    orienatation = 'L'  # Potrait and landscape, default =landscape
    def __init__(self, path):
        self.path = path
        self.font = None
        try:
            self.output = Image.open(path)
        except:
            raise ValueError('Unable to open specified image path')


def myprint(img, tot_noi, photo_size=[3.5, 4.5], paper_size=[6, 4], orienatation='L'):
    # Photo sizes:
    # 2.5 x 3.2 cm,
    # 2.5 x 3.5 cm.
    # 3.5 x 3.5 cm
    # 3.5 x 4.5 cm.
    # 5.1 x 5.1 cm,

    # Paper Sizes
    # 6 x 4 inch
    # 7 x 5 inch
    # 6 x 6 inch
    # 8 x 6 inch

    def inch2pxl(inp):
        return np.round(np.multiply(inp, 96))

    def cm2pxl(inp):
        return np.round(np.multiply(inp, 96 / 2.54))

    photo_size = cm2pxl(photo_size)
    photo_size = photo_size.astype(int)

    photo_size_row = photo_size[0]
    photo_size_col = photo_size[1]

    img = cv2.resize(img, (photo_size_row, photo_size_col))

    BLUE = [255, 255, 255]
    img_pad = cv2.copyMakeBorder(img.copy(), 2, 4, 2, 4, cv2.BORDER_CONSTANT, value=BLUE)

    # photo_size=np.add(photo_size,[1,1])

    if orienatation == 'P':
        paper_size.reverse()

    # if len(img.shape)==3:
    #     row,col,ch=img_pad.shape
    # else:
    #     row,col=img_pad.shape

    paper_size = inch2pxl(paper_size)  # paper_size in pixels

    pad_photo_size = img_pad.shape

    paper_size_row = paper_size[0]
    paper_size_col = paper_size[1]

    pad_photo_size_row = pad_photo_size[1]
    pad_photo_size_col = pad_photo_size[0]

    img_pad = cv2.resize(img_pad, (pad_photo_size_row, pad_photo_size_col))

    out_img = 255 * np.ones([paper_size_col, paper_size_row, 3])

    noi_col = np.int(np.floor(paper_size_row / pad_photo_size_row))
    noi_row = np.int(np.floor(paper_size_col / pad_photo_size_col))

    max_noi = noi_col * noi_row  # maximum number of images write as seprate function for return

    nos = 1

    next_row = 0

    for i in range(0, noi_row):
        next_col = 0
        for j in range(0, noi_col):
            if nos > tot_noi:
                break

            out_img[next_row:next_row + pad_photo_size_col, next_col:next_col + pad_photo_size_row] = img_pad
            next_col = next_col + pad_photo_size_row
            nos = nos + 1

            # print (next_col)
        next_row = next_row + pad_photo_size_col
        # print ('col is ',next_row)

    # out_img[0:170,0:132]=img

    return np.uint8(out_img)


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def crop_image(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('{0}/shape_predictor_68_face_landmarks.dat'.format(os.path.abspath(os.path.dirname(__file__))))

    len_face = []

    gray_rot = gray


    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        print(rect)

        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        (x, y, w, h) = rect_to_bb(rect)

        #	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #
        #	for (x, y) in shape:
        #		cv2.circle(image, (x, y), 1, (0, 0, 255), 5)

        (lStart, lEnd) = (42, 48)
        (rStart, rEnd) = (36, 42)
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        desiredLeftEye = (0.37, 0.4)
        desiredFaceWidth = 200
        desiredFaceHeight = 250

        desiredRightEyeX = 1.0 - desiredLeftEye[0]
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - desiredLeftEye[0])
        desiredDist *= desiredFaceWidth
        scale = desiredDist / dist

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = desiredFaceWidth * 0.5
        tY = desiredFaceHeight * desiredLeftEye[1]

        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        (w, h) = (desiredFaceWidth, desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        print_image = myprint(output, tot_noi, photo_size, paper_size, orienatation)

        # def water_mark(print_image):
        print_image = print_image[..., ::-1]
        wat_img = Image.fromarray(print_image, 'RGB')
        imOut = wat_img.convert("RGBA")
        txt = Image.new('RGBA', wat_img.size, (255, 255, 255, 0))

        # get a font
        fnt = ImageFont.truetype('arial.ttf', 40)
        # get a drawing context
        d = ImageDraw.Draw(txt)
        width, height = wat_img.size

        x = width / 16
        y = height / 16

        # draw text, half opacity
        d.text((x, y), "Cash thanittu upayogikado", font=fnt, fill=(255, 255, 255, 128))
        x = width / 4
        y = height / 2
        d.text((x, y), "Cash thanittu upayogikado", font=fnt, fill=(255, 255, 255, 128))
        txt = txt.rotate(30)
        final_img = Image.alpha_composite(imOut, txt)
        # final_img.show()
        return final_img

#
# final_img = water_mark(print_image)
# final_img.show()

# 	faceAligned = fa.align(image, gray, rect)
# 	cv2.imshow("Aligned", faceAligned)
# 	cv2.waitKey(0)



# for rot_ang in range(0, 4):
#     # image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#     rects_rot = detector(gray_rot, 1)
#     len_face.append(len(rects_rot))
#     gray_rot = cv2.rotate(gray_rot, cv2.ROTATE_90_CLOCKWISE)
#
# ang = len_face.index(max(len_face))
# if ang == 1:
#     gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
#     image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
# elif ang == 2:
#     gray = cv2.rotate(gray, cv2.ROTATE_180)
#     image = cv2.rotate(image, cv2.ROTATE_180)
# elif ang == 3:
#     gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

