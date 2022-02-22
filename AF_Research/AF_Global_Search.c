
void Global_search(unsigned int pos_begin, unsigned int pos_end){
  
  // 0. Init something
  unsigned long preTime = GetTimeStampMs();
  unsigned int pos_cur = pos_begin;

  AFMoveToPosition(pos_begin);

  unsigned int fv_cur = 0;
  KSAFGetFv(&fv_cur, 1);

  printf("*****************INIT_FV: %d %d\n", fv_cur, pos_begin);
  int temp_fv = fv_cur;
  int diff_fv = 0;

  AFMoveToPosition(pos_end);
  KSAFGetFv(&fv_cur, 0);
  printf("****************END_FV: %d %d\n", fv_cur, pos_end); 

  while (pos_begin < pos_cur && pos_cur < pos_end)
  {
      pos_cur = pos_cur + 1;
      AFMoveToPosition(pos_cur);
      KSAFGetFv(&fv_cur, 1);
      diff_fv = fv_cur - temp_fv;
    if (diff_fv > 0){
        temp_fv = fv_cur;
    }
  }
  int max_fv = temp_fv;
  while (pos_begin < pos_cur && pos_cur < pos_end)
  {
      pos_cur = pos_cur - 1;
      AFMoveToPosition(pos_cur);
      KSAFGetFv(&fv_cur, 1);
      diff_fv = fv_cur - max_fv;
    if (abs(diff_fv) < max_fv*0.01){
        printf("****************END_FV: %d %d\n", fv_cur, pos_cur); 
    }
  }

}


