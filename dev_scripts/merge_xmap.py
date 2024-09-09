import glob
import os
from tqdm import tqdm

# Define the directory containing the xmap files and the output file name
directory = '/Users/danielle/Downloads/output/contigs/exp_refineFinal1/alignmol/merge'
output_file = '/Users/danielle/Downloads/output/contigs/exp_refineFinal1/exp_refineFinal1_merged.xmap'

def merge_xmap_files(directory, output_file):
    # Define the regex pattern to find the xmap files
    pattern = os.path.join(directory, 'exp_refineFinal1_contig[0-9]*.xmap')
    files = sorted(glob.glob(pattern))

    # Initialize variables
    headers = []
    merged_lines = []
    current_id = 1

    # Process each file
    for file in tqdm(files):
        with open(file, 'r') as f:
            lines = f.readlines()
            if not headers:
                headers = lines[:10]  # Get the first 10 lines as headers
            for line in lines[10:]:
                # Split the line into columns
                columns = line.split()
                # Update the XmapEntryID (first column) with the current_id
                columns[0] = str(current_id)
                # Join the columns back into a single line
                merged_line = '\t'.join(columns)
                merged_lines.append(merged_line)
                current_id += 1

    # Write the merged content to the output file
    with open(output_file, 'w') as out_file:
        out_file.writelines(headers)
        out_file.write('\n'.join(merged_lines))
    print('Done merging')


def preview_file(file_path, n=200, head=True):
    # Open the file for reading
    with open(file_path, 'r') as file:
        # Read the first n lines
        if head:
            lines = file.readlines()[:n]
            preview_file_path = file_path.replace('.bnx', '_head.bnx')
        else:
            lines = file.readlines()[-n:]
    
    # Create the preview file name
            preview_file_path = file_path.replace('.bnx', '_tail.bnx')
    
    # Write the preview lines to the preview file
    with open(preview_file_path, 'w') as preview_file:
        preview_file.writelines(lines)
    
    print(f'Preview file created: {preview_file_path}')

def strip_genome_file(file_path):
    # open the file and save a new file that only contains lines from the original file that contain 'NC_'
    with open(file_path, 'r') as file:  # open the file
        lines = file.readlines()
    new_lines = [line for line in tqdm(lines) if 'NC_' in line]
    new_file_path = file_path.replace('.txt', '_chromosome_names.fna')
    with open(new_file_path, 'w') as new_file:
        new_file.writelines(new_lines)
    print('Done stripping genome file, new file at: ', new_file_path)
    

# Merge the xmap files
#merge_xmap_files(directory, output_file)
# preview_file('/Users/danielle/Downloads/data_T1_chip2_channels_swapped.bnx')
preview_file('/Users/danielle/Downloads/S278C_B_--_CAGOIS6NPM37DN6U_--_F1P1_--_2023-07-13T135051.142Z_RawMolecules.bnx')
# strip_genome_file('/Users/danielle/Downloads/data_combined_S288C_reference_sequence_R64-3-1_20210421_and_GCF_000001405.40_GRCh38.p14_genomic.fna')
# strip_genome_file('/Users/danielle/Downloads/data_GCF_000001405.40_GRCh38.p14_genomic.fna')


def remove_newlines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    new_lines = [line.strip() for line in tqdm(lines)]
    new_file_path = file_path.replace('.txt', '_no_newlines.txt')
    with open(new_file_path, 'w') as new_file:
        new_file.write(' '.join(new_lines))
    print('Done removing newlines, new file at: ', new_file_path)