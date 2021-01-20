from flask import Flask, render_template, Response
from imutils.video import VideoStream
from imutils.perspective import four_point_transform
from imutils import contours
from pytz import timezone
import datetime
import threading
import cv2
import time
import imutils
from gpiozero import PWMLED

lock = threading.Lock()

led0 = PWMLED(17)
led1 = PWMLED(23)

webcam0 = None
webcam1 = None
vs0 = None
vs1 = None
outputFrame0 = None
outputFrame1 = None

digits = [0,0,0,0]
rpm = 0.0
hz = 0.0

def get_frame0():
    global webcam0, vs0, outputFrame0, lock

    webcam0 = VideoStream(src=0).start()

    while True:
        vs0 = webcam0.read()

        with lock:
            outputFrame0 = vs0.copy()

def get_frame1():
    global webcam1, vs1, outputFrame1, lock

    webcam1 = VideoStream(src=2).start()

    while True:
        vs1 = webcam1.read()

        with lock:
            outputFrame1 = vs1.copy()

def gen0():
    global outputFrame0, lock
    
    while True:
        with lock:
            if outputFrame0 is None:
                continue

            (flag0, encodedImage0) = cv2.imencode(".jpg",outputFrame0)

            if not flag0:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage0) + b'\r\n')


def gen1():
    global outputFrame1, lock
    
    while True:
        with lock:
            if outputFrame1 is None:
                continue

            (flag1, encodedImage1) = cv2.imencode(".jpg",outputFrame1)

            if not flag1:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage1) + b'\r\n')

DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 1, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
}

def gen_digit():
    global outputFrame0, lock, digits, rpm, hz
    
    findDigits = True
    digitsX = 0
    digitsY = 0
    digitsW = 0
    digitsH = 0
    digit_count = 0

    while True:
        with lock:
            if outputFrame0 is None:
                continue

            crop = outputFrame0[150:260,150:320].copy()

            while findDigits:
                digit_count += 1
                #gray = cv2.cvtColor(outputFrame0[170:240,170:300], cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edged = cv2.Canny(blurred, 50, 200, 255)

                cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
                displayCnt = None
                for c in cnts:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    if len(approx) == 4:
                        displayCnt = approx
                        (digitsX, digitsY, digitsW, digitsH) = cv2.boundingRect(displayCnt)
                        findDigits = False
                        break

                if digit_count > 10:
                    #FIXME
                    digitsX = 24
                    digitsY = 22
                    digitsW = 124
                    digitsH = 62
                    findDigits = False
                    break

            #output = cv2.resize(four_point_transform(crop, displayCnt.reshape(4, 2)),dsize=(130,80),interpolation=cv2.INTER_LINEAR)
            output = cv2.resize(crop[digitsY:digitsY+digitsH,digitsX:digitsX+digitsW],dsize=(130,80),interpolation=cv2.INTER_LINEAR)
            warped = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

            #thresh = cv2.threshold(warped, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            thresh = cv2.bitwise_not(cv2.threshold(warped, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            digitCnts = []
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                if (w >= 10 and w <= 20) and (h >= 20 and h <= 50):
                    digitCnts.append(c)
                    #print(cv2.boundingRect(c))
            #digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

            #for c in digitCnts:
            #FIXME
            fixedCnts = (43,61,79,96)
            digits = []
            for c in fixedCnts:
                #(x, y, w, h) = cv2.boundingRect(c)
                (x, y, w, h) = (c, 12, 15, 30)
                roi = thresh[y:y + h, x:x + w]
                (roiH, roiW) = roi.shape
                #(dW, dH) = (int(roiW * 0.3), int(roiH * 0.2))
                (dW, dH) = (4,4)
                dHC = 2
                segments = [
                    ((dW, 0), (w-dW, dH)),	# top
                    ((0, dH), (dW, (h // 2)-dHC)),	# top-left
                    ((w - dW, dH), (w, (h // 2)-dHC)),	# top-right
                    ((dW, (h // 2) - dHC) , (w - dW, (h // 2) + dHC)), # center
                    ((0, (h // 2)+dHC), (dW, h-dH)),	# bottom-left
                    ((w - dW, (h // 2)+dHC), (w, h-dH)),	# bottom-right
                    ((dW, h - dH), (w-dW, h))	# bottom
                ]
                on = [0] * len(segments)

                for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                    segROI = roi[yA:yB, xA:xB]
                    total = cv2.countNonZero(segROI)
                    area = (xB - xA) * (yB - yA)
                    if float(area) > 0 and total / float(area) > 0.3:
                    #with light
                    #if float(area) > 0 and total / float(area) > 0.7:
                    #for night
                        on[i]= 1
                if tuple(on) in DIGITS_LOOKUP:
                    digit = DIGITS_LOOKUP[tuple(on)]
                else:
                    digit = 0
                    #print(x,tuple(on))
                digits.append(digit)
                #cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 255), 1)
                cv2.putText(output, str(digit), (x + 5, y + 45),cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)


            rpm = digits[0]*10 + digits[1]*1 + digits[2]*0.1 + digits[3]*0.01
            hz = rpm/60.0
            (flag_d, encodedImage_d) = cv2.imencode(".jpg",output)
            #(flag_d, encodedImage_d) = cv2.imencode(".jpg",thresh)

            if not flag_d:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage_d) + b'\r\n')


app = Flask(__name__)
@app.route('/')
def index():
    datestr = datetime.datetime.now().strftime("%Y/%m/%d")
    timestr = datetime.datetime.now().strftime("%H:%M")
    templateData = {'date':datestr,'time':timestr}
    return render_template('index.html',**templateData)

@app.route('/time_feed')
def time_feed():
    def get_time():
        yield datetime.datetime.now().astimezone(timezone('Asia/Seoul')).strftime("%Y/%m/%d %H:%M:%S")
    return Response(get_time(),mimetype='text')

@app.route('/video_feed0')
def video_feed0():
    return Response(gen0(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed1')
def video_feed1():
    return Response(gen1(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_digit0')
def video_digit0():
    return Response(gen_digit(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/rpm_feed')
def rpm_feed():
    def get_rpm():
        global rpm
        yield '{:.2f}'.format(rpm)
    return Response(get_rpm(),mimetype='text')
    
@app.route('/hz_feed')
def hz_feed():
    def hz_rpm():
        global hz
        return '{:.4f}'.format(hz) 
    return Response(hz_rpm(),mimetype='text')

@app.route('/<number>/<value>')
def led_control(number,value):
    if int(number) is 0:
        led1.value=float(value)/100.0
    else:
        led0.value=float(value)/100.0
    return (''), 204

            
if __name__ == "__main__":
    t1 = threading.Thread(target=get_frame1)
    t0 = threading.Thread(target=get_frame0)
    t0.daemon = True
    t1.daemon = True
    t0.start()
    t1.start()
    app.run(host='163.152.20.27', port=80, threaded=True, use_reloader=False)

webcam0.stop()
webcam1.stop()
