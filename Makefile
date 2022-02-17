

all:
	g++ main.cpp reconocedor.cpp utils.cpp -I/home/computacion/opencv/opencvi/include/opencv4 -L/home/computacion/opencv/opencvi/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_video -lopencv_videoio  -o vision.bin

run:
	./vision.bin