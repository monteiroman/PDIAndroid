package com.example.tiago.pdiandroid;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.widget.TextView;

/**
 * Created by tiago on 16/11/17.
 */

public class ShowText extends Activity{

    String txt;
    TextView showText;

    @Override
    public void onCreate(Bundle savedInstanceState){
        super.onCreate(savedInstanceState);
        setContentView(R.layout.output);

        final Intent intent = getIntent();
        txt = intent.getStringExtra("text");

        showText = (TextView) findViewById(R.id.outputText);

        showText.setText(txt);

    }

}
