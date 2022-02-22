/*
 * @Author: yuanht
 * @Date: 2022-02-21 11:39:54
 * @LastEditors: yuanht
 * @LastEditTime: 2022-02-21 19:44:07
 * @Description: cver
 */
void Goldensection_search(unsigned int pos_begin, unsigned int pos_end){
  
  // 0. Init something
  unsigned long preTime = GetTimeStampMs();
  unsigned int pos_cur = pos_begin;
  unsigned int pos_best = pos_begin;

  double alpha_idx = 0.382;
  double beta_idx = 0.618;

  unsigned int pos_t1 = pos_begin + round(alpha_idx * (pos_end - pos_begin));
  unsigned int pos_t2 = pos_end - round(beta_idx * (pos_end - pos_begin));
  AFMoveToPosition(pos_begin);

  unsigned int moving_win_idx = 0;
  unsigned int moving_win_sum = 0;
  unsigned int moving_win_avg = 0;
  unsigned int moving_win_pre = 0;
  unsigned int moving_win_max = 0;
  unsigned int fv_cur = 0;
  KSAFGetFv(&fv_cur, 1);

  printf("*****************INIT_FV: %d %d\n", fv_cur, pos_begin);
  int temp_fv = fv_cur;

  AFMoveToPosition(pos_end);
  KSAFGetFv(&fv_cur, 0);
  printf("****************END_FV: %d %d\n", fv_cur, pos_end); 

  int diff_step = pos_t2 - pos_t1;
  int diff_fv = fv_cur - temp_fv;

  while (diff_step > 10)
  {
    /* right > left */ 
    if (diff_fv >= 0){
      pos_end = pos_t2;
      pos_t2 = pos_t1;

      AFMoveToPosition(pos_t2);
      KSAFGetFv(&fv_cur, 1);
      temp_fv = fv_cur;
      pos_t1 = pos_begin + round(alpha_idx * (pos_end - pos_begin));
      AFMoveToPosition(pos_t1);
      KSAFGetFv(&fv_cur, 1);
      diff_step = pos_t2 - pos_t1;
      diff_fv = temp_fv - fv_cur;
      printf("****************END_FV1: %d %d %d\n", diff_step, pos_t1, pos_t2); 
    }
    else if (diff_fv < 0){

      pos_begin = pos_t1;
      pos_t1 = pos_t2;

      AFMoveToPosition(pos_t2);
      KSAFGetFv(&fv_cur, 1);
      temp_fv = fv_cur;
      pos_t2 = pos_end - round(beta_idx * (pos_end - pos_begin));
      AFMoveToPosition(pos_t2);
      KSAFGetFv(&fv_cur, 1);
      diff_step = pos_t2 - pos_t1;
      diff_fv = fv_cur - temp_fv;
      printf("****************END_FV2: %d %d %d\n", diff_step, pos_t1, pos_t2); 
    }

    else {
      break;
    }

  }
}
