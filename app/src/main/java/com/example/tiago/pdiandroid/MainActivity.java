package com.example.tiago.pdiandroid;

import android.app.Activity;

import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.util.SparseArray;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.TextView;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.barcode.Barcode;
import com.google.android.gms.vision.barcode.BarcodeDetector;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;
import org.opencv.videoio.VideoCapture;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import static org.opencv.core.CvType.CV_16S;
import static org.opencv.videoio.Videoio.CV_CAP_PROP_FRAME_HEIGHT;
import static org.opencv.videoio.Videoio.CV_CAP_PROP_FRAME_WIDTH;



/*__________________________________________________________//

Usar matchShapes



//__________________________________________________________*/

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2{

    private static final String TAG = "OCVSample::Activity";
    private static final int SUBMATWIDTH = 640;
    private static final int SUBMATHEIGHT = 480;
    private int  w, h;
    private CameraBridgeViewBase mOpenCvCameraView;
//    TextView tvName;
//    Scalar RED = new Scalar(255, 0, 0);
//    Scalar GREEN = new Scalar(0, 255, 0);
//    FeatureDetector detector;
//    DescriptorExtractor descriptor;
//    DescriptorMatcher matcher;
//    Mat descriptors2,descriptors1;
//    Mat img1;
//    MatOfKeyPoint keypoints1,keypoints2;
//    Mat grad_X = new Mat();
//    Mat grad_Y = new Mat();
//    Mat abs_grad = new Mat();
//    Mat thres_out = new Mat();
//    Mat morph_out = new Mat();
    private Mat myInputFrame = new Mat();
    private Mat warpedFrame = new Mat();
    private Mat outputFrame = new Mat();
    private Mat myInputFrameGray = new Mat();
    private Mat sub = new Mat();

//    List<MatOfPoint> squares = new ArrayList<MatOfPoint>();
//    int thresh = 50, N = 11;
    Mat smallerImg, gray, gray0;

//    VideoCapture mCamera;
    TextView txtView;
    Bitmap myBitmap;




    static {
        if (!OpenCVLoader.initDebug())
            Log.d("ERROR", "Unable to load OpenCV");
        else
            Log.d("SUCCESS", "OpenCV loaded");
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    try {
                        initializeOpenCVDependencies();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    private void initializeOpenCVDependencies() throws IOException {
        mOpenCvCameraView.enableView();
    }


    public MainActivity() {

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {

        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.layout);

        //mCamera = new VideoCapture();
        //mCamera.set(CV_CAP_PROP_FRAME_WIDTH, 640);
        //mCamera.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

//        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_java_surface_view);
        //mOpenCvCameraView.setMaxFrameSize(1024, 860);

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        //tvName = (TextView) findViewById(R.id.text1);
        //txtView = (TextView) findViewById(R.id.txtContent);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        w = width;
        h = height;
        Log.i(TAG, "Width: " + w);
        Log.i(TAG, "Height: " + h);
    }

    public void onCameraViewStopped() {
    }


    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        myInputFrame = inputFrame.rgba();
//        aInputFrame.copyTo(bInputFrame);
//        myInputFrame.copyTo(outputFrame);
        myInputFrame.copyTo(warpedFrame);


//        bInputFrame = aInputFrame.clone();
//        cInputFrame = aInputFrame.clone();
//        Mat aux;

        //1-------------If i want to detect the barcode
        //aux = detectBarCode(aInputFrame.clone());
        //findSquares(aux, squares);

        //2-------------If I don't want to detect the barcode
        //findSquares(inputFrame.gray(), squares);

        //Imgproc.drawContours(aInputFrame, squares, -1, new Scalar(0,255,0), 3);

        //return aux;


        //3-------------If i want to use google's android vision API's
        /*
        myBitmap = Bitmap.createBitmap(aInputFrame.cols(),  aInputFrame.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(aInputFrame, myBitmap);

        BarcodeDetector detector = new BarcodeDetector.Builder(getApplicationContext())
                        .setBarcodeFormats(Barcode.DATA_MATRIX | Barcode.QR_CODE)
                        .build();
        if(!detector.isOperational()){
            txtView.setText("Could not set up the detector!");
        }

        Frame frame = new Frame.Builder().setBitmap(myBitmap).build();
        SparseArray<Barcode> barcodes = detector.detect(frame);


        if(barcodes.size()==0){
            //txtView.setText("No se detectó un codigo");
            Log.d(TAG, "No se econtró codigo");
        }else {
            Barcode thisCode = barcodes.valueAt(0);
            //txtView.setText(thisCode.rawValue);
            Log.d(TAG, thisCode.rawValue);
            final Intent intent = new Intent(this, ShowText.class);
            intent.putExtra("text", thisCode.rawValue);
            startActivity(intent);
        }
        */

//        //4-------------Another way to detect squares and warp them.
//        Imgproc.cvtColor(aInputFrame, aInputFrame, Imgproc.COLOR_BGR2GRAY);
//
//        //convert the image to black and white does (8 bit)
//        Imgproc.Canny(aInputFrame, aInputFrame, 50, 50);
//
//        //apply gaussian blur to smoothen lines of dots
//        Imgproc.GaussianBlur(aInputFrame, aInputFrame, new  org.opencv.core.Size(5, 5), 5);
//
//        //find the contours
//        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
//        Imgproc.findContours(aInputFrame, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

//        -----------------------------------------------------------------------------
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        myFindContours(myInputFrame.clone(), contours);



        if (!contours.isEmpty()) {

            double maxArea = -1;
            MatOfPoint temp_contour;
            MatOfPoint2f approxCurve = new MatOfPoint2f();

            for (int idx = 0; idx < contours.size(); idx++) {
                temp_contour = contours.get(idx);
                double contourarea = Imgproc.contourArea(temp_contour);

                // compare this contour to the previous largest contour found
                if (contourarea > maxArea) {
                    // check if this contour is a square
                    MatOfPoint2f new_mat = new MatOfPoint2f(temp_contour.toArray());
                    int contourSize = (int) temp_contour.total();
                    MatOfPoint2f approxCurve_temp = new MatOfPoint2f();
                    Imgproc.approxPolyDP(new_mat, approxCurve_temp, contourSize * 0.05, true);

                    if (approxCurve_temp.total() == 4) {
                        maxArea = contourarea;
                        approxCurve = approxCurve_temp;
                    }
                }
            }

            if (maxArea>0) {
                //Imgproc.cvtColor(imgSource, imgSource, Imgproc.COLOR_BayerBG2RGB);
                //Mat sourceImage = Highgui.imread(fileName, Highgui.CV_LOAD_IMAGE_UNCHANGED);
                double[] temp_double;
                temp_double = approxCurve.get(0, 0);
                Point p1 = new Point(temp_double[0], temp_double[1]);
                // Core.circle(imgSource,p1,55,new Scalar(0,0,255));
                // Imgproc.warpAffine(sourceImage, dummy, rotImage,sourceImage.size());
                temp_double = approxCurve.get(1, 0);
                Point p2 = new Point(temp_double[0], temp_double[1]);
                // Core.circle(imgSource,p2,150,new Scalar(255,255,255));
                temp_double = approxCurve.get(2, 0);
                Point p3 = new Point(temp_double[0], temp_double[1]);
                // Core.circle(imgSource,p3,200,new Scalar(255,0,0));
                temp_double = approxCurve.get(3, 0);
                Point p4 = new Point(temp_double[0], temp_double[1]);
                // Core.circle(imgSource,p4,100,new Scalar(0,0,255));
                List<Point> source = new ArrayList<Point>();
                source.add(p1);
                source.add(p2);
                source.add(p3);
                source.add(p4);
                Mat startM = Converters.vector_Point2f_to_Mat(source);

                warpedFrame = warp(myInputFrame.clone(), startM, w, h);
            }
        }


        detectBarcode (warpedFrame);
//        detectBarcode (myInputFrame);

        outputFrame = subMat(myInputFrame, warpedFrame);

        return outputFrame;

//        -----------------------------------------------------------------------------
    }

//    Mat detectBarCode (Mat aInputFrame){
//
//        int ddepth = CV_16S;
//        int scale = 1;
//        int delta = 0;
//
//        org.opencv.core.Size s = new Size(5,5);
//        Imgproc.GaussianBlur(aInputFrame, aInputFrame, s, 0, 0);
//        Imgproc.cvtColor(aInputFrame, aInputFrame, Imgproc.COLOR_RGB2GRAY);
//
//        Imgproc.Scharr(aInputFrame, grad_X, ddepth, 1, 0, scale, delta);
//        Imgproc.Scharr(aInputFrame, grad_Y, ddepth, 0, 1, scale, delta);
//
//        Core.subtract(grad_X, grad_Y, abs_grad);
//        Core.convertScaleAbs(abs_grad, abs_grad);
//        //Core.convertScaleAbs(grad_X, abs_grad);
//
//        org.opencv.core.Size s2 = new Size(5,5);
//        Imgproc.GaussianBlur(abs_grad, abs_grad, s2, 0, 0);
//        Imgproc.threshold(abs_grad, thres_out, 230, 255, 0);
//
//
//        Imgproc.morphologyEx(thres_out, morph_out, Imgproc.MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(30,1)));
//
//        Imgproc.erode(morph_out, morph_out, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2,2)));
//        Imgproc.dilate(morph_out, aInputFrame, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1,1)));
//
//        return aInputFrame;
//    }


//    // returns sequence of squares detected on the image.
//    // the sequence is stored in the specified memory storage
//    void findSquares( Mat image, List<MatOfPoint> squares )
//    {
//        squares.clear();
//
//        smallerImg=new Mat(new Size(image.width()/2, image.height()/2),image.type());
//
//        gray=new Mat(image.size(),image.type());
//
//        gray0=new Mat(image.size(),CvType.CV_8U);
//
//        // down-scale and upscale the image to filter out the noise
//        Imgproc.pyrDown(image, smallerImg, smallerImg.size());
//        Imgproc.pyrUp(smallerImg, image, image.size());
//
//        gray = image;
//
//        //Cany removed... Didn't work so well
//
//        Imgproc.threshold(gray, gray0, 90, 255, Imgproc.THRESH_BINARY);
//
//        List<MatOfPoint> contours=new ArrayList<MatOfPoint>();
//
//        // find contours and store them all as a list
//        Imgproc.findContours(gray0, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
//
//        MatOfPoint approx=new MatOfPoint();
//
//        // test each contour
//        for( int i = 0; i < contours.size(); i++ )
//        {
//
//            // approximate contour with accuracy proportional
//            // to the contour perimeter
//            approx = approxPolyDP(contours.get(i),  Imgproc.arcLength(new MatOfPoint2f(contours.get(i).toArray()), true)*0.02, true);
//
//
//            // square contours should have 4 vertices after approximation
//            // relatively large area (to filter out noisy contours)
//            // and be convex.
//            // Note: absolute value of an area is used because
//            // area may be positive or negative - in accordance with the
//            // contour orientation
//
//            if( approx.toArray().length == 4 &&
//                    Math.abs(Imgproc.contourArea(approx)) > 1000 &&
//                    Imgproc.isContourConvex(approx) )
//            {
//                double maxCosine = 0;
//
//                for( int j = 2; j < 5; j++ )
//                {
//                    // find the maximum cosine of the angle between joint edges
//                    double cosine = Math.abs(angle(approx.toArray()[j%4], approx.toArray()[j-2], approx.toArray()[j-1]));
//                    maxCosine = Math.max(maxCosine, cosine);
//                }
//
//                // if cosines of all angles are small
//                // (all angles are ~90 degree) then write quandrange
//                // vertices to resultant sequence
//                if( maxCosine < 0.3 )
//                    squares.add(approx);
//            }
//
//        }
//    }

//    // helper function:
//    // finds a cosine of angle between vectors
//    // from pt0->pt1 and from pt0->pt2
//    double angle( Point pt1, Point pt2, Point pt0 ) {
//        double dx1 = pt1.x - pt0.x;
//        double dy1 = pt1.y - pt0.y;
//        double dx2 = pt2.x - pt0.x;
//        double dy2 = pt2.y - pt0.y;
//        return (dx1*dx2 + dy1*dy2)/Math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
//    }

//   void extractChannel(Mat source, Mat out, int channelNum) {
//        List<Mat> sourceChannels=new ArrayList<Mat>();
//        List<Mat> outChannel=new ArrayList<Mat>();
//
//        Core.split(source, sourceChannels);
//
//        outChannel.add(new Mat(sourceChannels.get(0).size(),sourceChannels.get(0).type()));
//
//        Core.mixChannels(sourceChannels, outChannel, new MatOfInt(channelNum,0));
//
//        Core.merge(outChannel, out);
//    }

//    MatOfPoint approxPolyDP(MatOfPoint curve, double epsilon, boolean closed) {
//        MatOfPoint2f tempMat=new MatOfPoint2f();
//
//        Imgproc.approxPolyDP(new MatOfPoint2f(curve.toArray()), tempMat, epsilon, closed);
//
//        return new MatOfPoint(tempMat.toArray());
//    }


    Mat subMat (Mat bigMat, Mat smallMat){
        Rect roi = new Rect(10, 10, SUBMATWIDTH, SUBMATHEIGHT);
        sub = bigMat.submat(roi);
        Imgproc.resize(smallMat, smallMat, new Size(SUBMATWIDTH, SUBMATHEIGHT));
        smallMat.copyTo(sub);
        return bigMat;
    }

    public static Mat warp(Mat inputMat, Mat startM, int resultWidth, int resultHeight) {

//        int resultWidth = SUBMATWIDTH;
//        int resultHeight = SUBMATHEIGHT;
//        int resultWidth = 1200;
//        int resultHeight = 680;

        Point ocvPOut4 = new Point(0, 0);
        Point ocvPOut3 = new Point(0, resultHeight);
        Point ocvPOut2 = new Point(resultWidth, resultHeight);
        Point ocvPOut1 = new Point(resultWidth, 0);

        if (inputMat.height() > inputMat.width()) {
            // int temp = resultWidth;
            // resultWidth = resultHeight;
            // resultHeight = temp;

            ocvPOut1 = new Point(0, 0);
            ocvPOut4 = new Point(0, resultHeight);
            ocvPOut3 = new Point(resultWidth, resultHeight);
            ocvPOut2 = new Point(resultWidth, 0);
        }

        Mat outputMat = new Mat(resultWidth, resultHeight, CvType.CV_8UC4);

        List<Point> dest = new ArrayList<Point>();
        dest.add(ocvPOut1);
        dest.add(ocvPOut2);
        dest.add(ocvPOut3);
        dest.add(ocvPOut4);

        Mat endM = Converters.vector_Point2f_to_Mat(dest);

        Mat perspectiveTransform = Imgproc.getPerspectiveTransform(startM, endM);

        Imgproc.warpPerspective(inputMat, outputMat, perspectiveTransform, new Size(resultWidth, resultHeight), Imgproc.INTER_CUBIC);

        return outputMat;
    }

    public void myFindContours (Mat image, List<MatOfPoint> contours){
        contours.clear();

        Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2GRAY);

        //convert the image to black and white does (8 bit)
        Imgproc.Canny(image, image, 50, 50);

        //apply gaussian blur to smoothen lines of dots
        Imgproc.GaussianBlur(image, image, new  org.opencv.core.Size(5, 5), 5);

        //find the contours
        Imgproc.findContours(image, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
    }

    public void detectBarcode (Mat image){
        myBitmap = Bitmap.createBitmap(image.cols(),  image.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(image, myBitmap);

        BarcodeDetector detector = new BarcodeDetector.Builder(getApplicationContext())
                .setBarcodeFormats(Barcode.DATA_MATRIX | Barcode.PDF417)
                .build();
        if(!detector.isOperational()){
            txtView.setText("Could not set up the detector!");
        }

        Frame frame = new Frame.Builder().setBitmap(myBitmap).build();
        SparseArray<Barcode> barcodes = detector.detect(frame);


        if(barcodes.size()==0){
            //txtView.setText("No se detecto un codigo");
            Log.d(TAG, "No se econtró codigo");
        }else {
            Barcode thisCode = barcodes.valueAt(0);
            //txtView.setText(thisCode.rawValue);
            Log.d(TAG, thisCode.rawValue);
            final Intent intent = new Intent(this, ShowText.class);
            intent.putExtra("text", thisCode.rawValue);
            startActivity(intent);
        }

    }

}

