def random_crop_image(im, boxes, ignored_boxes, shape, ensure_gt=False):
    im_h, im_w = im.shape[:2]

    if boxes and ensure_gt:

        # step 1: random choice a gt
        gt_idx = np.random.randint(len(boxes))
        x1_selected, y1_selected, w_selected, h_selected, _ = boxes[gt_idx]
        x2_selected = x1_selected + w_selected - 1
        y2_selected = y1_selected + h_selected - 1

        # step 2: resize im if face too large
        selected_fpn_idx = np.random.choice(
            np.arange(config.fpn_k_min, config.fpn_k_max + 1))
        base_resize_ratio = (2 ** selected_fpn_idx) * config.anchor_base_size

        min_anchor_size = np.sqrt(
            (base_resize_ratio ** 2) *
            np.array(config.anchor_scales).min() / 2
        )
        max_anchor_size = np.sqrt(
            (base_resize_ratio ** 2) *
            min(np.array(config.anchor_scales).max(), 1) * 2
        )
        max_anchor_size = min(max_anchor_size, max(config.image_shape[:2]) * 1.2)

        shortest_edge = min(w_selected, h_selected)
        longest_edge = max(w_selected, h_selected)

        min_resize_ratio = min_anchor_size / max(1., float(shortest_edge))
        max_resize_ratio = max_anchor_size / max(1., float(longest_edge))

        if min_resize_ratio < 2.5 and min_resize_ratio < max_resize_ratio:
            resize_ratio = np.random.uniform(low=min_resize_ratio, high=min(2.5, max_resize_ratio))
        else:
            resize_ratio = np.random.uniform(low=1, high=2.5)
            if np.random.randint(2) == 1:
                resize_ratio = 1. / resize_ratio

        # im_h, im_w = int(resize_ratio * im_h), int(resize_ratio * im_w)
        # im = imgproc.resize_preserve_aspect_ratio(im, (im_h, im_w))
        im = cv2.resize(im, (0, 0), fx=resize_ratio, fy=resize_ratio)
        im_h, im_w = im.shape[:2]

        # step 3: revise bboxes
        for bb_idx, bb in enumerate(boxes):
            x, y, w, h, _ = bb
            boxes[bb_idx][0] = x * resize_ratio
            boxes[bb_idx][1] = y * resize_ratio
            boxes[bb_idx][2] = w * resize_ratio
            boxes[bb_idx][3] = h * resize_ratio

        # boxes = revise_bboxes(boxes, (im_h, im_w))

        x1_selected = x1_selected * resize_ratio
        y1_selected = y1_selected * resize_ratio
        x2_selected = min(im_w - 1, x2_selected * resize_ratio)
        y2_selected = min(im_h - 1, y2_selected * resize_ratio)
        w_selected = x2_selected - x1_selected + 1
        h_selected = y2_selected - y1_selected + 1

        for bb_idx, bb in enumerate(ignored_boxes):
            x, y, w, h, _ = bb
            ignored_boxes[bb_idx][0] = x * resize_ratio
            ignored_boxes[bb_idx][1] = y * resize_ratio
            ignored_boxes[bb_idx][2] = w * resize_ratio
            ignored_boxes[bb_idx][3] = h * resize_ratio

        # ignored_boxes = revise_bboxes(ignored_boxes, (im_h, im_w))

        try:
            crop_x1_min = max(0, x2_selected - shape[1])
            crop_x1_max = min(x1_selected, max(0, im_w - shape[1]))
            crop_x1 = np.random.randint(crop_x1_min, crop_x1_max + 1)
            crop_y1_min = max(0, y2_selected - shape[0])
            crop_y1_max = min(y1_selected, max(0, im_h - shape[0]))
            crop_y1 = np.random.randint(crop_y1_min, crop_y1_max + 1)
        except:
            crop_x1 = np.random.randint(0, max(1, im_w - shape[1] + 0))
            crop_y1 = np.random.randint(0, max(1, im_h - shape[0] + 0))

    else:
        crop_x1 = np.random.randint(0, max(1, im_w - shape[1] + 0))
        crop_y1 = np.random.randint(0, max(1, im_h - shape[0] + 0))

    crop_x2 = min(im_w - 1, crop_x1 + shape[1] - 1)
    crop_y2 = min(im_h - 1, crop_y1 + shape[0] - 1)

    crop_w = crop_x2 - crop_x1 + 1
    crop_h = crop_y2 - crop_y1 + 1

    im = im[crop_y1:(crop_y2 + 1), crop_x1:(crop_x2 + 1)]

    final_boxes = []
    final_ignored_boxes = []

    for bb in boxes:
        x, y, w, h, c = bb

        x1 = max(x - crop_x1, 0)
        y1 = max(y - crop_y1, 0)
        x2 = min(x + w - 1 - crop_x1, crop_w - 1)
        y2 = min(y + h - 1 - crop_y1, crop_h - 1)

        if x1 >= x2 or y1 >= y2:
            continue

        if not crop_x1 <= x + w / 2 <= crop_x2 or not crop_y1 <= y + h / 2 <= crop_y2:
            final_ignored_boxes.append([x1, y1, x2, y2, config.anchor_ignore_label])
        else:
            final_boxes.append([x1, y1, x2, y2, c])

    for bb in ignored_boxes:
        x, y, w, h, c = bb

        x1 = max(x - crop_x1, 0)
        y1 = max(y - crop_y1, 0)
        x2 = min(x + w - 1 - crop_x1, crop_w - 1)
        y2 = min(y + h - 1 - crop_y1, crop_h - 1)

        if x1 >= x2 or y1 >= y2:
            continue

        final_ignored_boxes.append([x1, y1, x2, y2, c])

    im, (ph, pw) = imgproc.pad_image_to_shape(im, shape, return_padding=True)
    for i, _ in enumerate(final_boxes):
        final_boxes[i][0] += pw
        final_boxes[i][1] += ph
        final_boxes[i][2] += pw
        final_boxes[i][3] += ph
    for i, _ in enumerate(final_ignored_boxes):
        final_ignored_boxes[i][0] += pw
        final_ignored_boxes[i][1] += ph
        final_ignored_boxes[i][2] += pw
        final_ignored_boxes[i][3] += ph

    return im, final_boxes, final_ignored_boxes