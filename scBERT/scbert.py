#change form
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import sys
import os

def align_genes_and_save(target_path, ref_path, output_path):
    print(f"--- 작업 시작: {target_path} ---")

    # 1. 파일 존재 여부 확인
    if not os.path.exists(target_path):
        print(f"Error: 변환할 파일을 찾을 수 없습니다: {target_path}")
        return
    if not os.path.exists(ref_path):
        print(f"Error: 기준 파일을 찾을 수 없습니다: {ref_path}")
        return

    # 2. 데이터 로드
    adata = sc.read_h5ad(target_path)
    adata_ref = sc.read_h5ad(ref_path)

    # [추가] 중복 이름 해결 (에러 방지)
    if not adata.obs_names.is_unique:
        print("중복된 세포 이름이 발견되어 고유하게 수정합니다.")
        adata.obs_names_make_unique()
    if not adata.var_names.is_unique:
        print("중복된 유전자 이름이 발견되어 고유하게 수정합니다.")
        adata.var_names_make_unique()

    # 3. 데이터 행렬 재배열
    print("데이터 행렬 재배열 중 (reindex)...")
    # sparse matrix를 dataframe으로 변환 후 재배열
    data_df = adata.to_df().reindex(columns=adata_ref.var_names, fill_value=0)

    # 4. 유전자 정보(var) 보존
    # 원본에 있던 gene_ids 등을 panglao 순서에 맞게 가져옵니다.
    new_var = adata.var.reindex(adata_ref.var_names)

    # 5. 새로운 AnnData 생성
    new_adata = sc.AnnData(
        X=csr_matrix(data_df.values),
        obs=adata.obs.copy(),
        var=new_var,
        uns=adata.uns.copy()
    )

    # 6. 데이터 타입 최적화
    new_adata.X = new_adata.X.astype('float32')

    # 7. 저장
    new_adata.write_h5ad(output_path)

    print(f"✅ 변환 완료!")
    print(f"결과 구조: {new_adata.shape}")
    print(f"저장 위치: {output_path}\n")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("사용법: python scbert.py <source_file.h5ad> <reference_file.h5ad> <output_file.h5ad>")
    else:
        src = sys.argv[1]
        ref = sys.argv[2]
        out = sys.argv[3]

        # 함수 실행 (이제 경로 문자열을 직접 넘겨줍니다)
        align_genes_and_save(src, ref, out)
