import os
import re
from pathlib import Path

def get_avi_files(folder_path):
    """폴더에서 모든 .avi 파일의 이름을 가져옵니다."""
    try:
        folder = Path(folder_path)
        if not folder.exists():
            print(f"폴더가 존재하지 않습니다: {folder_path}")
            return set()
        
        avi_files = {f.name for f in folder.glob("*.avi")}
        return avi_files
    except Exception as e:
        print(f"폴더 읽기 오류 ({folder_path}): {e}")
        return set()

def remove_rtmo_bytetrack_overlay(filename):
    """파일명에서 _rtmo*_bytetrack_overlay 패턴을 제거합니다."""
    # _rtmo로 시작하고 _bytetrack_overlay로 끝나는 패턴을 찾아서 제거
    # 예: "1XFiS6Lt_1_rtmo_bytetrack_overlay.avi" -> "1XFiS6Lt_1.avi"
    pattern = r'_rtmo.*?_bytetrack_overlay'
    cleaned_name = re.sub(pattern, '', filename)
    return cleaned_name

def compare_video_folders():
    # 폴더 경로 설정
    original_folder = "/aivanas/raw/surveillance/action/violence/action_recognition/data/RWF-2000/train/Fight"
    processed_folder = "/workspace/mmpose/output/RWF-2000/train/Fight"
    
    print("비디오 파일 비교 분석 시작...")
    print(f"원본 폴더: {original_folder}")
    print(f"처리된 폴더: {processed_folder}")
    print("-" * 80)
    
    # 각 폴더에서 .avi 파일 목록 가져오기
    original_files = get_avi_files(original_folder)
    processed_files = get_avi_files(processed_folder)
    
    print(f"원본 폴더 .avi 파일 개수: {len(original_files)}")
    print(f"처리된 폴더 .avi 파일 개수: {len(processed_files)}")
    print("-" * 80)
    
    # 처리된 파일명에서 rtmo_bytetrack_overlay 패턴 제거
    cleaned_processed_files = set()
    for filename in processed_files:
        cleaned_name = remove_rtmo_bytetrack_overlay(filename)
        cleaned_processed_files.add(cleaned_name)
    
    print(f"정리된 처리 파일명 개수: {len(cleaned_processed_files)}")
    print("-" * 80)
    
    # 원본에는 있지만 처리된 폴더에는 없는 파일들
    missing_in_processed = original_files - cleaned_processed_files
    
    # 처리된 폴더에는 있지만 원본에는 없는 파일들
    extra_in_processed = cleaned_processed_files - original_files
    
    # 결과 출력
    print("📋 분석 결과:")
    print("=" * 80)
    
    print(f"🔍 ORIGINAL에는 있는데, PROCESS에는 없는 파일들 ({len(missing_in_processed)}개):")
    if missing_in_processed:
        for filename in sorted(missing_in_processed):
            print(f"   - {filename}")
    else:
        print("   ✅ 없음 (모든 original 파일이 process에 존재)")
    print()
    
    print(f"🔍 PROCESS에는 있는데, ORIGINAL에는 없는 파일들 ({len(extra_in_processed)}개):")
    if extra_in_processed:
        for filename in sorted(extra_in_processed):
            print(f"   - {filename}")
    else:
        print("   ✅ 없음 (process에 추가 파일 없음)")
    print()
    
    # 매핑 정보 출력 (선택사항)
    print("🔗 처리된 파일과 원본 파일 매핑 예시:")
    print("-" * 50)
    sample_count = 0
    for processed_file in sorted(processed_files):
        if sample_count >= 5:  # 처음 5개만 보여주기
            break
        cleaned_name = remove_rtmo_bytetrack_overlay(processed_file)
        if cleaned_name in original_files:
            print(f"   {processed_file} -> {cleaned_name}")
            sample_count += 1
    
    if len(processed_files) > 5:
        print(f"   ... (총 {len(processed_files)}개 파일)")

if __name__ == "__main__":
    compare_video_folders()