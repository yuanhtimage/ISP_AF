
void Fibonacci_search(unsigned int pos_begin, unsigned int pos_end){
  unsigned int a = 0;
  unsigned int b = 500;
  unsigned int x1;
  unsigned int x2; 
  unsigned int iterFS = 20;
  double fibArr[iterFS];
  unsigned int fv_cur = 0;

  //构造斐波那契数列
	fibArr[1] = fibArr[0] = 1;
	for (int i=2;i<iterFS;i++)
	{
		fibArr[i] = fibArr[i - 1] + fibArr[i - 2];
	}

  x1 = a + fibArr[iterFS - 3] / fibArr[iterFS-1] * (b-a);
  x2 = a + fibArr[iterFS - 2] / fibArr[iterFS-1] * (b-a);

  int t = iterFS;
  int i = 0;
  printf("INIT: %d %d\n", x1,x2);

  unsigned int temp = 350;
  //&& x1-a>10 && b-x2>10 
  while (t > 3 && x1-a>30 && b-x2>30 )
  {
    //printf("-------------RUNNING----------: NO. %d, begin_diff: %d , end_diff: %d\n", i, x1-a,  b-x2);
    /*
    自行定义以下两个函数。
    电机移动函数：AFMoveToPosition(x1+temp)
    计算图像清晰度评价函数：KSAFGetFv(&fv_cur, 1)
    */
    i += 1;
    AFMoveToPosition(x1+temp);    
    KSAFGetFv(&fv_cur, 1);
    unsigned int fv1_cur = fv_cur;
    AFMoveToPosition(x2+temp);
    KSAFGetFv(&fv_cur, 1);
    unsigned int fv2_cur = fv_cur;
    
    if (fv1_cur < fv2_cur){
        a = x1;
        x1 = x2;
        x2 = a + round(fibArr[t - 2] / fibArr[t-1] * (b-a)); 
    }

    else if (fv1_cur > fv2_cur)
    {
      b = x2;
      x2 = x1;
      x1 = a + round(fibArr[t - 3] / fibArr[t-1] * (b-a));
    }
    
    else{
        a = x1;
        b = x2;
        x1 = a + round(fibArr[t - 3] / fibArr[t-1] * (b-a));
        x2 = a + round(fibArr[t - 2] / fibArr[t-1] * (b-a));
    }

    t -= 1;
    printf("-------------RUNNING----------: NO. %d, begin: %d , end: %d\n", i, x1,  x2);
    printf("-------------RUNNING----------: fv1: %d , fv2: %d\n",  fv1_cur,  fv2_cur); 

  }
}