#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FS_BYTE_LEN 		8
#define FS_BYTE_MAX 		0xFF

#define PUT_4BYTES(__addr, __data) \
{\
    *(__addr) = (uchar)(__data & 0xFF);\
    *((__addr) + 1) = (uchar)(((__data) >> FS_BYTE_LEN) & 0xFF);\
    *((__addr) + 2) = (uchar)(((__data) >> (FS_BYTE_LEN * 2)) & 0xFF);\
    *((__addr) + 3) = (uchar)(((__data) >> (FS_BYTE_LEN * 3)) & 0xFF);\
}

#define PUT_2BYTES(__addr, __data) \
{\
    *(__addr) = (uchar)(__data & 0xFF);\
    *((__addr) + 1) = (uchar)(((__data) >> FS_BYTE_LEN) & 0xFF);\
}

#define GET_4BYTES(__addr, __data) \
{\
    __data = 0;\
    __data |= *(__addr);\
    __data |= (*((__addr) + 1)) << (FS_BYTE_LEN * 1);\
    __data |= (*((__addr) + 2)) << (FS_BYTE_LEN * 2);\
    __data |= (*((__addr) + 3)) << (FS_BYTE_LEN * 3);\
}

#define GET_2BYTES(__addr, __data) \
{\
    __data = 0;\
    __data |= *(__addr);\
    __data |= (*((__addr) + 1)) << (FS_BYTE_LEN * 1);\
}

#define FS_TIME_OFFEST          22
#define FS_CREAT_TIME_OFFEST    26
#define FS_SIZE_OFFEST          30
#define FS_OP_READ              0


#define FS_GSYS_OP_TIME         0
#define FS_GSYS_OP_FSIZE        1
#define FS_NODE_MAX             1024

__device__ __managed__ u32 gtime = 0;

__device__ void fs_set_super_storage_index(FileSystem *fs, u32 storage_index, uchar bit);
__device__ uchar *fs_get_block_address(FileSystem *fs, u32 node_index);
__device__ int fs_get_storage_address(FileSystem *fs, u32 storage_index);
__device__ void fs_set_time(FileSystem *fs, u32 node_index, u32 time);
__device__ void fs_set_creat_time(FileSystem *fs, u32 node_index, u32 createTime);
__device__ void fs_set_size(FileSystem *fs, u32 node_index, u32 size);
__device__ void fs_set_storage_index(FileSystem *fs, u32 node_index, u32 storage_index);
__device__ void fs_set_used(FileSystem *fs, u32 node_index, uchar use_bit);
__device__ void fs_set_filename(FileSystem *fs, u32 node_index, char *filename);
__device__ u32 fs_get_time(FileSystem *fs, u32 node_index);
__device__ u32 fs_get_creat_time(FileSystem *fs, u32 node_index);
__device__ u32 fs_get_size(FileSystem *fs, u32 node_index);
__device__ u32 fs_get_storage_index(FileSystem* fs, u32 node_index);
__device__ void fs_get_filename(FileSystem *fs, u32 node_index, char *buf);
__device__ uchar fs_get_used(FileSystem *fs, u32 node_index);
__device__ int fs_match_filename(FileSystem* fs, char *file_name, int node_index);
__device__ int fs_find_file_node_index(FileSystem *fs, char *file_name);
__device__ u32 fs_find_storage_index(FileSystem *fs);
__device__ void fs_set_block_entry(FileSystem* fs, u32 node_index, char* filename,
                    u32 storage_index, u32 time, u32 createTime, u32 size, uchar use_bit);
__device__ u32 fs_find_file_node_storage_index(FileSystem* fs, u32 storage_index);
__device__ void fs_file_merge(FileSystem* fs, u32 node_index);
__device__ int fs_open_new(FileSystem *fs, char *s);
__device__ int fs_open_exist(FileSystem *fs, int node_index, char *s);

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS, u32 *FILE_TIME)
{
	// init variables
  	fs->volume = volume;

  	// init constants
  	fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  	fs->FCB_SIZE = FCB_SIZE;
  	fs->FCB_ENTRIES = FCB_ENTRIES;
  	fs->STORAGE_SIZE = VOLUME_SIZE;
  	fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  	fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  	fs->MAX_FILE_NUM = MAX_FILE_NUM;
  	fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  	fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;
  	fs->FILE_TIME = FILE_TIME;
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	int node_index;

    //printf("fs_open op %d %s\n", op, s);

	(*fs->FILE_TIME)++;

	node_index = fs_find_file_node_index(fs, s);
	if (op == FS_OP_READ) {
        //printf("fs_open read %d\n", node_index);
		return node_index;
	}

	if (node_index == fs->FCB_ENTRIES) {
        node_index = fs_open_new(fs, s);
        //printf("fs_open_new %d\n", node_index);
		return node_index;
	} else {
        node_index = fs_open_exist(fs, node_index, s);
        //printf("fs_open_exist %d\n", node_index);
        return node_index;
	}
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
    int j;

    //printf("fs_read size %d\n", size);

	*(fs->FILE_TIME) += 1;

	u32 storage_index = fs_get_storage_index(fs, fp);
	u32 storage_offest = fs_get_storage_address(fs, storage_index);

	for (j = 0; j < size; j++) {
		output[j] = *(fs->volume + storage_offest + j);
	}

   // printf("fs_read end\n");

    return;
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
    int blcok, i;

    //printf("fs_write size %d\n", size);

	*(fs->FILE_TIME)++;
    fs_set_time(fs, fp, *fs->FILE_TIME);
    fs_set_size(fs, fp, size);

	u32 storage_index = fs_get_storage_index(fs, fp);
    u32 storage_offest = fs_get_storage_address(fs, storage_index);
	for (i = 0; i < size; i++) {
		*(fs->volume + storage_offest + i) = input[i];
	}

    int mode = size / fs->STORAGE_BLOCK_SIZE;
    int left = size % fs->STORAGE_BLOCK_SIZE;
	if (left != 0) {
        blcok = mode + 1;
    } else {
        blcok = mode;
    }

	for (i = 0; i < blcok; i++) {
		fs_set_super_storage_index(fs, storage_index + i, 1);
	}

    //printf("fs_write end\n");
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
    int i, j;
	u32 tmp_node_index[FS_NODE_MAX];
	int node_cnt = 0;
    u32 tmp;
    char buf[256];

	*(fs->FILE_TIME)++;

    //printf("op %d\n", op);

    if (op != FS_GSYS_OP_TIME && op != FS_GSYS_OP_FSIZE) {
		printf("Wrong op %d code!\n", op);
        return;
	}

	for (i = 0; i < fs->FCB_ENTRIES; i++) {
		if (fs_get_used(fs, i)) {
			tmp_node_index[node_cnt] = i;
			node_cnt++;
		}
	}

    //printf("node_cnt %d!\n", node_cnt);

	if (op == FS_GSYS_OP_TIME) {
		for (i = 0; i < node_cnt - 1; i++) {
			for (j = 0; j < node_cnt - 1 - i; j++) {
				if (fs_get_time(fs, tmp_node_index[j]) < fs_get_time(fs, tmp_node_index[j + 1])) {
	                tmp = tmp_node_index[j];
					tmp_node_index[j] = tmp_node_index[j + 1];
					tmp_node_index[j + 1] = tmp;
				}
			}
		}

		printf("=== Sort by modified time ===\n");
		for (i = 0; i < node_cnt; i++) {
            fs_get_filename(fs, tmp_node_index[i], buf);
			printf("%s\n", buf);
		}

        return;
	}

    //file size
	for (i = 0; i < node_cnt - 1; i++) {
		for (j = 0; j < node_cnt - 1 - i; j++) {
			if (fs_get_size(fs, tmp_node_index[j]) < fs_get_size(fs, tmp_node_index[j + 1])) {
				tmp = tmp_node_index[j];
				tmp_node_index[j] = tmp_node_index[j + 1];
				tmp_node_index[j + 1] = tmp;
			} else if (fs_get_size(fs, tmp_node_index[j]) == fs_get_size(fs, tmp_node_index[j + 1])) {
				if (fs_get_time(fs, tmp_node_index[j]) > fs_get_time(fs, tmp_node_index[j + 1])) {
			        tmp = tmp_node_index[j];
					tmp_node_index[j] = tmp_node_index[j + 1];
					tmp_node_index[j + 1] = tmp;
				}
			}
		}
	}
	printf("=== Sort by file size ===\n");
	for (i = 0; i < node_cnt; i++) {
        fs_get_filename(fs, tmp_node_index[i], buf);
		printf("%s %d\n", buf, fs_get_size(fs, tmp_node_index[i]));
	}

    return;
}

__device__ void fs_gsys(FileSystem* fs, int op, char* s)
{
	*(fs->FILE_TIME)++;

	if (op != 2) {
        printf("Invalid op %d ??\n", op);
		return;
    }

    //printf("op %d file %s\n", op, s);
	u32 node_index = fs_find_file_node_index(fs, s);
	if (node_index == fs->FCB_ENTRIES) {
		printf("not find??");
		return;
	}

	fs_file_merge(fs, node_index);

    return;
}

__device__ void fs_set_super_storage_index(FileSystem *fs, u32 storage_index, uchar bit)
{
	u32 mode = storage_index / FS_BYTE_LEN;
	u32 left = storage_index % FS_BYTE_LEN;

	if (bit) {
        *(fs->volume + mode) |= (0x1 << left);
    } else {
		*(fs->volume + mode) &= (~(0x1 << left));
	}
}

__device__ uchar *fs_get_block_address(FileSystem *fs, u32 node_index)
{
	return fs->volume + fs->SUPERBLOCK_SIZE + node_index * fs->FCB_SIZE;
}

__device__ int fs_get_storage_address(FileSystem *fs, u32 storage_index)
{
	return fs->FILE_BASE_ADDRESS + storage_index * fs->STORAGE_BLOCK_SIZE;
}

__device__ void fs_set_time(FileSystem *fs, u32 node_index, u32 time)
{
    uchar *addr = fs_get_block_address(fs, node_index);
    PUT_4BYTES(addr + FS_TIME_OFFEST, time);
}

__device__ void fs_set_creat_time(FileSystem *fs, u32 node_index, u32 createTime)
{
    uchar *addr = fs_get_block_address(fs, node_index);
    PUT_4BYTES(addr + FS_CREAT_TIME_OFFEST, createTime);
}

__device__ void fs_set_size(FileSystem *fs, u32 node_index, u32 size)
{
    uchar *addr = fs_get_block_address(fs, node_index);
    PUT_2BYTES(addr + FS_SIZE_OFFEST, size);
}

__device__ void fs_set_storage_index(FileSystem *fs, u32 node_index, u32 storage_index)
{
    uchar *addr = fs_get_block_address(fs, node_index);
	*(addr + 20) = (uchar)(storage_index & 0xFF);
	*(addr + 21) &= 0x80;
	*(addr + 21) |= (uchar)((storage_index >> FS_BYTE_LEN) & 0x7F);
}

__device__ void fs_set_used(FileSystem *fs, u32 node_index, uchar use_bit)
{
    uchar *addr = fs_get_block_address(fs, node_index);
    if (use_bit == 0) {
        *(addr + 21) &= 0x7F;
    } else {
        *(addr + 21) |= 0x80;
    }

    return;
}

__device__ void fs_set_filename(FileSystem *fs, u32 node_index, char *filename)
{
    int i;
    uchar *addr = fs_get_block_address(fs, node_index);

	for (i = 0; i < fs->MAX_FILENAME_SIZE; i++) {
		*(addr + i) = filename[i];
		if (filename[i] == 0) {
            break;
        }
	}

    return;
}

__device__ u32 fs_get_time(FileSystem *fs, u32 node_index)
{
    uchar *addr = fs_get_block_address(fs, node_index);
    u32 data;

    GET_4BYTES(addr + FS_TIME_OFFEST, data);

    return data;
}

__device__ u32 fs_get_creat_time(FileSystem *fs, u32 node_index)
{
    uchar *addr = fs_get_block_address(fs, node_index);
    u32 data;

    GET_4BYTES(addr + FS_CREAT_TIME_OFFEST, data);

    return data;
}

__device__ u32 fs_get_size(FileSystem *fs, u32 node_index)
{
    uchar *addr = fs_get_block_address(fs, node_index);
    u32 data;

    GET_2BYTES(addr + FS_SIZE_OFFEST, data);

    return data;
}

__device__ u32 fs_get_storage_index(FileSystem* fs, u32 node_index)
{
	u32 data = 0;
    uchar *addr = fs_get_block_address(fs, node_index);

	data |= *(addr + 20);
	data |= (*(addr + 21) & 0x7F) << FS_BYTE_LEN;

	return data;
}

__device__ void fs_get_filename(FileSystem *fs, u32 node_index, char *buf)
{
	int i;
    uchar *addr = fs_get_block_address(fs, node_index);

	for (i = 0; i < fs->MAX_FILENAME_SIZE; i++) {
		buf[i] = *(addr + i);
		if (*(addr + i) == 0) {
            break;
        }
	}

	return;
}

__device__ uchar fs_get_used(FileSystem *fs, u32 node_index)
{
	uchar *addr = fs_get_block_address(fs, node_index);

	return (*(addr + 21) >> 7);
}

//-------------------------------------------------------------------------
__device__ int fs_match_filename(FileSystem* fs, char *file_name, int node_index)
{
	int i;
	uchar *addr = fs_get_block_address(fs, node_index);

	//scan MAX FILENAME LEN
	for (i = 0; i < fs->MAX_FILENAME_SIZE; i++) {
		if (file_name[i] == '\0') {
			return 1;
		}

		if (*(addr + i) != file_name[i]) {
			return 0;
		}
	}

	return 0;
}

__device__ int fs_find_file_node_index(FileSystem *fs, char *file_name)
{
	int i;

	for (i = 0; i < fs->FCB_ENTRIES; i++) {
		if (fs_get_used(fs, i) == 0) {
			continue;
		}

		if (fs_match_filename(fs, file_name, i)) {
			return i;
		}
	}

	return fs->FCB_ENTRIES;
}

__device__ u32 fs_find_storage_index(FileSystem *fs)
{
    int i, j;

	for (i = 0; i < fs->SUPERBLOCK_SIZE; i++) {
		if (*(fs->volume + i) == FS_BYTE_MAX) {
            continue;
        }

		for (j = 0; j < FS_BYTE_LEN; j++) {
			if (!((*(fs->volume + i) >> j) & 0x1)) {
				return (i * FS_BYTE_LEN + j);
			}
		}
	}

	return fs->SUPERBLOCK_SIZE * FS_BYTE_LEN;
}

__device__ void fs_set_block_entry(FileSystem* fs, u32 node_index, char* filename,
                    u32 storage_index, u32 time, u32 createTime, u32 size, uchar use_bit)
{
    uchar *addr = fs_get_block_address(fs, node_index);
    fs_set_creat_time(fs, node_index, createTime);
    fs_set_time(fs, node_index, time);
    fs_set_size(fs, node_index, size);
	fs_set_filename(fs, node_index, filename);
	fs_set_storage_index(fs, node_index, storage_index);
	fs_set_used(fs, node_index, use_bit);

    return;
}

__device__ u32 fs_find_file_node_storage_index(FileSystem* fs, u32 storage_index)
{
    int i;
    u32 tmp_sindex;

	for (i = 0; i < fs->FCB_ENTRIES; i++) {
		if (fs_get_used(fs, i) == 0) {
			continue;
		}

        tmp_sindex = fs_get_storage_index(fs, i);
        if (tmp_sindex == storage_index) {
			return i;
		}
	}

	return fs->FCB_ENTRIES;
}

__device__ void fs_file_merge(FileSystem* fs, u32 node_index)
{
    u32 i, j;
	u32 mleft;
	u32 msize = fs_get_size(fs, node_index);
    int tmp_node_index;
    u32 block_addr_offest, block_addr_left_offest;
    u32 block_node_index, begin_addr, end_addr;

	//printf("%s-%d, msize %u\n", __FUNCTION__, __LINE__, msize);
	
    if (msize % 32) {
        mleft = msize >> 5 + 1;
    } else {
        mleft = msize >> 5;
    }
	//printf("%s-%d, mleft %u\n", __FUNCTION__, __LINE__, mleft);
	
    block_node_index = fs_get_storage_index(fs, node_index);
	//printf("%s-%d\n", __FUNCTION__, __LINE__);
    begin_addr = fs_get_storage_address(fs, block_node_index);
	//printf("%s-%d\n", __FUNCTION__, __LINE__);
    end_addr = fs_find_storage_index(fs);

	fs_set_used(fs, node_index, 0);

	//printf("%s-%d, block_node_index %u, begin_addr %u, end_addr %u - mleft %u = %u\n", __FUNCTION__, __LINE__, 
	//	block_node_index, begin_addr, end_addr, mleft, end_addr-mleft);
	
	for (i = block_node_index; i < end_addr-mleft; i++) {
        block_addr_offest = begin_addr + i * fs->STORAGE_BLOCK_SIZE;
        block_addr_left_offest = begin_addr + (i + mleft) * fs->STORAGE_BLOCK_SIZE;
		for (j = 0; j < 32; j++) {
			*(fs->volume + block_addr_offest + j) = *(fs->volume + block_addr_left_offest + j);
		}
	
        tmp_node_index = fs_find_file_node_storage_index(fs, i+mleft);
		if (tmp_node_index != fs->FCB_ENTRIES) {
			fs_set_storage_index(fs, tmp_node_index, i);
		}
	}
	//printf("%s-%d\n", __FUNCTION__, __LINE__);
	for (i = block_node_index; i < end_addr - mleft; i++) {
		fs_set_super_storage_index(fs, i, 1);
	}
	//printf("%s-%d\n", __FUNCTION__, __LINE__);
	for (i = 0; i < mleft; i++) {
		fs_set_super_storage_index(fs, end_addr-i-1, 0);
	}
	//printf("%s-%d\n", __FUNCTION__, __LINE__);
    return;
}

__device__ int fs_open_new(FileSystem *fs, char *s)
{
    int i, node_index, bit_index;

    for (i = 0; i < fs->FCB_ENTRIES; i++) {
		if (fs_get_used(fs, i) == 0) {
			node_index = i;
		    bit_index = fs_find_storage_index(fs);
			fs_set_block_entry(fs, node_index, s, bit_index, 0, *(fs->FILE_TIME), 0, 1);
			return node_index;
		}
	}

    return 0;
}

__device__ int fs_open_exist(FileSystem *fs, int node_index, char *s)
{
    u32 bit_index;
    uchar *addr = fs_get_block_address(fs, node_index);
    u32 createTime;

	//printf("fs_open_exist begin\n");
	
    fs_file_merge(fs, node_index);
	//printf("fs_file_merge end\n");
	
    bit_index = fs_find_storage_index(fs);
    createTime = fs_get_creat_time(fs, node_index);
	fs_set_block_entry(fs, node_index, s, bit_index, 0, createTime, 0, 1);

	//printf("fs_open_exist begin\n");
	
	return node_index;
}

