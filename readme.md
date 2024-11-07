# Qshing 탐지 모델 개발

## 실행

- batch size는 gpu varm에 맞게 변경해야 합니다. 배치사이즈 32 기준 22.4gb를 사용합니다.
- [데이터 다운로드](https://drive.google.com/file/d/10oGckwjm4M1bnzibx1qlGLDPhYa2TaUB/view?usp=sharing) 후 data 폴더를 만들어서 다운로드한 파일을 넣어주세요.

```shell
python main.py -dn part.db --batch_size 1
```
