package com.example.testcamera;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.MSER;
import org.opencv.features2d.ORB;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.SIFT;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.io.IOException;

import static org.opencv.core.Core.*;
import static org.opencv.imgproc.Imgproc.circle;

public class MainActivity extends Activity {
    private static final int CAMERA_REQUEST = 1888;
    private ImageView imageView;
    private static final int MY_CAMERA_PERMISSION_CODE = 100;

    static {
        if (!OpenCVLoader.initDebug())
            Log.d("ERROR", "Unable to load OpenCV");
        else
            Log.d("SUCCESS", "OpenCV loaded");
    }
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        this.imageView = (ImageView)this.findViewById(R.id.imageView1);
        Button photoButton = (Button) this.findViewById(R.id.button1);
        photoButton.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View v)
            {
                if (ContextCompat.checkSelfPermission(v.getContext(), Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
                {
                    ActivityCompat.requestPermissions( (Activity) v.getContext(), new String[]{Manifest.permission.CAMERA}, MY_CAMERA_PERMISSION_CODE);
                }
                else
                {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, CAMERA_REQUEST);
                }
            }
        });
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults)
    {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == MY_CAMERA_PERMISSION_CODE)
        {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED)
            {
                Toast.makeText(this, "camera permission granted", Toast.LENGTH_LONG).show();
                Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent, CAMERA_REQUEST);
            }
            else
            {
                Toast.makeText(this, "camera permission denied", Toast.LENGTH_LONG).show();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        if (requestCode == CAMERA_REQUEST && resultCode == Activity.RESULT_OK)
        {
          Bitmap templateBmp = BitmapFactory.decodeResource(getResources(), R.drawable.template2);
            Mat templateC =  new Mat (templateBmp.getWidth(), templateBmp.getHeight(), CvType.CV_8UC1);
            Utils.bitmapToMat(templateBmp, templateC);
            Mat template = new Mat();

            Bitmap allBmp = BitmapFactory.decodeResource(getResources(), R.drawable.all);
            Mat allC =  new Mat (allBmp.getWidth(), allBmp.getHeight(), CvType.CV_8UC1);
            Utils.bitmapToMat(templateBmp, template);

            templateC.convertTo(template, CvType.CV_8UC1);

            Log.d("SUCCESS", "recognizion started" + (template.type()==CvType.CV_8UC1) );
            try {
                Mat m = featureMatching();
                Bitmap bm = Bitmap.createBitmap(m.cols(), m.rows(),Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(m, bm);
                Log.d("SUCCESS", "recognizion finished");
                imageView.setImageBitmap(bm);
            } catch (IOException e) {
                e.printStackTrace();
            }


        }
    }

    private Mat Recognize( Mat img1, Mat img2){

        Mat out = new Mat();
      //  Mat im1 = new Mat();
    //    Mat im2 = new Mat();
            MatOfDMatch emptyMatch = new MatOfDMatch();
            MatOfKeyPoint emptyKey1 = new MatOfKeyPoint();
            MatOfKeyPoint emptyKey2 = new MatOfKeyPoint();
            Features2d.drawMatches(img2, emptyKey1, img1, emptyKey2, emptyMatch, out);

        Imgproc.cvtColor(out, out, Imgproc.COLOR_BGR2RGB);
        Imgproc.putText(out, "Frame", new Point(img1.width() / 2,30), 1, 2, new Scalar(0,255,255),3);
        Imgproc.putText(out, "Match", new Point(img1.width() + img2.width() / 2,30), 1, 2, new Scalar(255,0,0),3);
        return out;
    }

    private Mat FindTemplate() throws IOException {

        Mat img = Utils.loadResource(this, R.drawable.all2, CvType.CV_8UC4);
        Mat templ = Utils.loadResource(this, R.drawable.template2, CvType.CV_8UC4);

        int match_method = Imgproc.TM_CCORR_NORMED;

        Log.d("SUCCESS", "create result matrix");

        // / Create the result matrix
        int result_cols = img.cols() - templ.cols() + 1;
        int result_rows = img.rows() - templ.rows() + 1;
        Mat result = new Mat(result_rows, result_cols, img.type());

        Log.d("SUCCESS", "do matching and normalize") ;
        // / Do the Matching and Normalize
        Imgproc.matchTemplate(img, templ, result, match_method);
        normalize(result, result, 0, 1, NORM_MINMAX, -1, new Mat());

        Log.d("SUCCESS", "localizing");
        // / Localizing the best match with minMaxLoc
        MinMaxLocResult mmr = minMaxLoc(result);

        Point matchLoc;
        if (match_method == Imgproc.TM_SQDIFF || match_method == Imgproc.TM_SQDIFF_NORMED) {
            matchLoc = mmr.minLoc;
        } else {
            matchLoc = mmr.maxLoc;
        }
        Log.d("SUCCESS", "show what you got");
        // / Show me what you got
        Imgproc.rectangle(img, matchLoc, new Point(matchLoc.x + templ.cols(),
                matchLoc.y + templ.rows()), new Scalar(0, 255, 0));
        return img;
    }

    public Mat featureMatching() throws IOException {
        Mat input = Utils.loadResource(this, R.drawable.all2, CvType.CV_8UC4);
        Mat logo = Utils.loadResource(this, R.drawable.template2, CvType.CV_8UC4);
        SIFT detector = SIFT.create(4, 3);
    //    FastFeatureDetector detector = FastFeatureDetector.create();



        MatOfKeyPoint logoKeypoint = new MatOfKeyPoint();
        MatOfKeyPoint inputKeypoint = new MatOfKeyPoint();
        detector.detect(logo, logoKeypoint);
        detector.detect(input, inputKeypoint);

        Mat logoDes = new Mat();
        Mat inputDes = new Mat();
        detector.compute(logo, logoKeypoint, logoDes);
        detector.compute(input, inputKeypoint, inputDes);

        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
        MatOfDMatch matches = new MatOfDMatch();
        matcher.match(logoDes, inputDes, matches);
        int blockSize = 2;
        int apertureSize = 3;
        double k = 0.04;
        Mat imgMatches = new Mat();
       Features2d.drawMatches(logo, logoKeypoint, input, inputKeypoint, matches, imgMatches);
   //  Features2d.drawKeypoints(logo, logoKeypoint, imgMatches);
 //       Imgproc.cornerHarris(logo, imgMatches, blockSize, apertureSize, k);4

     //   findCorners(logo,imgMatches );
        Log.d("SUCCESS", "finally");
        return imgMatches;
    }

    private static Mat findCorners(Mat image, Mat energy) {

        Mat idx    = new Mat();
        Imgproc.cornerHarris(image, energy, 20, 9, 0.1);

        // Corner-search:

        int minDistance = 16;
        MinMaxLocResult minMaxLoc = minMaxLoc(
                energy.submat(20, energy.rows() - 20, 20, energy.rows() - 20));
        float thr = (float)minMaxLoc.maxVal / 4;

        Mat tmp = energy.reshape(1, 1);
        sortIdx(tmp, idx, 16); // 16 = CV_SORT_EVERY_ROW | CV_SORT_DESCENDING

        int[] idxArray = new int[idx.cols()];
        idx.get(0, 0, idxArray);
        float[] energyArray = new float[idx.cols()];
        energy.get(0, 0, energyArray);

        int n = 0;
        for (int p : idxArray) {
            if (energyArray[p] == -1) continue;
            if (energyArray[p] < thr) break;
            n++;

            int x = p % image.cols();
            int y = p / image.cols();

            // Exclude a disk around this corner from potential future candidates
            int u0 = Math.max(x - minDistance, 0) - x;
            int u1 = Math.min(x + minDistance, image.cols() - 1) - x;
            int v0 = Math.max(y - minDistance, 0) - y;
            int v1 = Math.min(y + minDistance, image.rows() - 1) - y;
            for (int v = v0; v <= v1; v++)
                for (int u = u0; u <= u1; u++)
                    if (u * u + v * v <= minDistance * minDistance)
                        energyArray[p + u + v * image.cols()] = -1;

            // A corner is found!
            circle(image, new Point(x, y), minDistance / 2, new Scalar(255, 255, 255), 1);
            circle(energy, new Point(x, y), minDistance / 2, new Scalar(minMaxLoc.maxVal, minMaxLoc.maxVal, minMaxLoc.maxVal), 1);
        }
        System.out.println("nCorners: " + n);

        // Rescale energy image for display purpose only

        multiply(energy, new Scalar(255.0 / minMaxLoc.maxVal), energy);

//      return image;
        return energy;
    }
}