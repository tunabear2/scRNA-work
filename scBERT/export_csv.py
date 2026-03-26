import scanpy as sc
import pandas as pd
import sys
import os

def export_to_csv(input_h5ad, output_csv):
    print(f"--- CSV 추출 시작: {input_h5ad} ---")

    if not os.path.exists(input_h5ad):
        print(f"Error: 파일을 찾을 수 없습니다: {input_h5ad}")
        return

    # 1. 데이터 로드
    adata = sc.read_h5ad(input_h5ad)

    # 2. 유전자 발현 데이터프레임 생성
    print("유전자 발현 행렬 변환 중...")
    matrix_df = adata.to_df()

    # 3. Cell Type 정보 찾기
    possible_cols = ['predicted_celltype', 'celltype', 'cell_type', 'leiden']
    found_col = None

    for col in possible_cols:
        if col in adata.obs.columns:
            found_col = col
            break

    if found_col:
        print(f"'{found_col}' 컬럼을 데이터의 맨 앞으로 배치합니다.")
        # [핵심] insert 함수를 사용하여 0번 인덱스(맨 앞)에 컬럼을 추가합니다.
        matrix_df.insert(0, 'cell_type_label', adata.obs[found_col])
    else:
        print("Warning: Cell Type 정보를 찾지 못했습니다. 유전자 데이터만 추출합니다.")

    # 4. CSV 저장
    # index=True로 설정하여 세포 바코드(index)도 함께 저장합니다.
    print(f"CSV 저장 중: {output_csv}")
    matrix_df.to_csv(output_csv)

    print(f"✅ 추출 완료! 파일 위치: {output_csv}")
    print(f"최종 데이터 크기: {matrix_df.shape}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("사용법: python export_csv.py <input_file.h5ad> <output_file.csv>")
    else:
        infile = sys.argv[1]
        outfile = sys.argv[2]
        export_to_csv(infile, outfile)
