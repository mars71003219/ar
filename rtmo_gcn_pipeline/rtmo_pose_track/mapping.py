import os
from pathlib import Path

def remove_postfix(filename, extension):
    """파일명에서 확장자와 postfix를 제거하여 베이스명을 얻습니다."""
    # 확장자 제거
    base_name = filename.replace(f'.{extension}', '')
    
    if extension == 'avi':
        # AVI 파일에서 _rtmo_bytetrack_overlay 제거
        base_name = base_name.replace('_rtmo_bytetrack_overlay', '')
    elif extension == 'pkl':
        # PKL 파일에서 _rtmo_bytetrack_pose 제거
        base_name = base_name.replace('_rtmo_bytetrack_pose', '')
    
    return base_name

def get_files_by_extension(folder_path, extension):
    """특정 확장자의 파일들을 가져와서 postfix를 제거한 베이스명을 반환합니다."""
    try:
        folder = Path(folder_path)
        if not folder.exists():
            print(f"폴더가 존재하지 않습니다: {folder_path}")
            return set()
        
        files = set()
        for f in folder.glob(f"*.{extension}"):
            base_name = remove_postfix(f.name, extension)
            files.add(base_name)
        
        return files
    except Exception as e:
        print(f"폴더 읽기 오류 ({folder_path}): {e}")
        return set()

def get_full_filenames_by_extension(folder_path, extension):
    """특정 확장자의 전체 파일명들과 베이스명 매핑을 가져옵니다."""
    try:
        folder = Path(folder_path)
        if not folder.exists():
            print(f"폴더가 존재하지 않습니다: {folder_path}")
            return {}, []
        
        files = []
        basename_to_fullname = {}
        for f in folder.glob(f"*.{extension}"):
            base_name = remove_postfix(f.name, extension)
            files.append(f.name)
            basename_to_fullname[base_name] = f.name
        
        return basename_to_fullname, sorted(files)
    except Exception as e:
        print(f"폴더 읽기 오류 ({folder_path}): {e}")
        return {}, []

def check_avi_pkl_mapping():
    # 폴더 경로 설정
    processed_folder = "/workspace/mmpose/output/RWF-2000/train/Fight"
    
    print("AVI-PKL 파일 매핑 검사 시작...")
    print(f"검사 폴더: {processed_folder}")
    print("AVI postfix: _rtmo_bytetrack_overlay")
    print("PKL postfix: _rtmo_bytetrack_pose")
    print("-" * 80)
    
    # 각 확장자별 파일 목록 가져오기 (postfix 제거한 베이스명)
    avi_basenames = get_files_by_extension(processed_folder, "avi")
    pkl_basenames = get_files_by_extension(processed_folder, "pkl")
    
    # 전체 파일명과 베이스명 매핑도 가져오기
    avi_mapping, avi_fullnames = get_full_filenames_by_extension(processed_folder, "avi")
    pkl_mapping, pkl_fullnames = get_full_filenames_by_extension(processed_folder, "pkl")
    
    print(f"AVI 파일 개수: {len(avi_basenames)}")
    print(f"PKL 파일 개수: {len(pkl_basenames)}")
    print("-" * 80)
    
    # 매핑 안되는 파일들 찾기
    avi_without_pkl = avi_basenames - pkl_basenames  # AVI는 있는데 PKL이 없는 것
    pkl_without_avi = pkl_basenames - avi_basenames  # PKL은 있는데 AVI가 없는 것
    
    # 결과 출력
    print("📋 매핑 검사 결과:")
    print("=" * 80)
    
    print(f"🔍 AVI는 있는데 PKL이 없는 파일들 ({len(avi_without_pkl)}개):")
    if avi_without_pkl:
        for basename in sorted(avi_without_pkl):
            avi_file = avi_mapping.get(basename, f"{basename}_rtmo_bytetrack_overlay.avi")
            expected_pkl = f"{basename}_rtmo_bytetrack_pose.pkl"
            print(f"   - {avi_file}")
            print(f"     (대응하는 {expected_pkl} 없음)")
    else:
        print("   ✅ 없음 (모든 AVI 파일에 대응하는 PKL 파일 존재)")
    print()
    
    print(f"🔍 PKL은 있는데 AVI가 없는 파일들 ({len(pkl_without_avi)}개):")
    if pkl_without_avi:
        for basename in sorted(pkl_without_avi):
            pkl_file = pkl_mapping.get(basename, f"{basename}_rtmo_bytetrack_pose.pkl")
            expected_avi = f"{basename}_rtmo_bytetrack_overlay.avi"
            print(f"   - {pkl_file}")
            print(f"     (대응하는 {expected_avi} 없음)")
    else:
        print("   ✅ 없음 (모든 PKL 파일에 대응하는 AVI 파일 존재)")
    print()
    
    # 매핑 통계
    matched_files = avi_basenames & pkl_basenames
    print(f"📊 매핑 통계:")
    print(f"   • 정상 매핑된 파일 쌍: {len(matched_files)}개")
    print(f"   • 매핑 안된 AVI 파일: {len(avi_without_pkl)}개")
    print(f"   • 매핑 안된 PKL 파일: {len(pkl_without_avi)}개")
    print(f"   • 전체 AVI 파일: {len(avi_basenames)}개")
    print(f"   • 전체 PKL 파일: {len(pkl_basenames)}개")
    print()
    
    # 매핑 완성도 계산
    if len(avi_basenames) > 0:
        avi_mapping_rate = (len(matched_files) / len(avi_basenames)) * 100
        print(f"   • AVI→PKL 매핑 완성도: {avi_mapping_rate:.1f}%")
    
    if len(pkl_basenames) > 0:
        pkl_mapping_rate = (len(matched_files) / len(pkl_basenames)) * 100
        print(f"   • PKL→AVI 매핑 완성도: {pkl_mapping_rate:.1f}%")
    
    # 샘플 매핑 예시 출력
    print("\n🔗 정상 매핑 예시:")
    print("-" * 50)
    sample_count = 0
    for basename in sorted(matched_files):
        if sample_count >= 3:
            break
        avi_file = avi_mapping.get(basename, f"{basename}_rtmo_bytetrack_overlay.avi")
        pkl_file = pkl_mapping.get(basename, f"{basename}_rtmo_bytetrack_pose.pkl")
        print(f"   베이스명: {basename}")
        print(f"   ├─ AVI: {avi_file}")
        print(f"   └─ PKL: {pkl_file}")
        print()
        sample_count += 1
    
    if len(matched_files) > 3:
        print(f"   ... (총 {len(matched_files)}개 매핑된 쌍)")
    
    # 전체 파일명 샘플 출력
    print("\n📁 전체 파일 예시:")
    print("-" * 50)
    print("AVI 파일:")
    for i, filename in enumerate(avi_fullnames[:3]):
        base_name = remove_postfix(filename, 'avi')
        print(f"   {filename} → 베이스명: {base_name}")
    if len(avi_fullnames) > 3:
        print(f"   ... (총 {len(avi_fullnames)}개)")
    
    print("\nPKL 파일:")
    for i, filename in enumerate(pkl_fullnames[:3]):
        base_name = remove_postfix(filename, 'pkl')
        print(f"   {filename} → 베이스명: {base_name}")
    if len(pkl_fullnames) > 3:
        print(f"   ... (총 {len(pkl_fullnames)}개)")

if __name__ == "__main__":
    check_avi_pkl_mapping()