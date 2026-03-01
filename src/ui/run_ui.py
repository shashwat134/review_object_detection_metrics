import os

import src.utils.converter as converter
import src.utils.general_utils as general_utils
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QShortcut
from PyQt5.QtGui import QKeySequence
from src.evaluators.coco_evaluator import get_coco_summary
from src.evaluators.pascal_voc_evaluator import (get_pascalvoc_metrics, plot_precision_recall_curve,
                                                 plot_precision_recall_curves)
from src.ui.details import Details_Dialog
from src.ui.main_ui import Ui_Dialog as Main_UI
from src.ui.results import Results_Dialog
from src.ui.splash import Splash_Dialog
from src.utils.enumerators import BBFormat, BBType, CoordinatesType


class Main_Dialog(QMainWindow, Main_UI):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.current_directory = os.path.dirname(os.path.realpath(__file__))
        # Define error msg dialog
        self.msgBox = QMessageBox()
        # Define details dialog
        self.dialog_statistics = Details_Dialog()
        # Define results dialog
        self.dialog_results = Results_Dialog()

        # Default values
        self.dir_annotations_gt = None
        self.dir_images_gt = None
        self.filepath_classes_gt = None
        self.dir_dets = None
        self.filepath_classes_det = None
        self.dir_save_results = None

        # --- Set the dialog background color to blue ---
        # A light blue tone is used so black labels / text stay readable.
        # We style both the QMainWindow and its central widget so the blue
        # background is visible regardless of platform / Qt style.
        blue_stylesheet = "background-color: #cfe8ff;"
        self.setStyleSheet("QMainWindow { %s }" % blue_stylesheet)
        if self.centralWidget() is not None:
            self.centralWidget().setStyleSheet(blue_stylesheet)
        else:
            # Fallback: paint via palette if no central widget was set.
            palette = self.palette()
            palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#cfe8ff"))
            self.setPalette(palette)
            self.setAutoFillBackground(True)

        # --- "Clear All" button near the top of the dialog ---
        # Placed at the top-right so it is easy to reach without covering
        # the existing "Ground truth" header label.
        if not hasattr(self, 'btn_clear_all') or self.btn_clear_all is None:
            parent_widget = self.centralWidget() if self.centralWidget() is not None else self
            self.btn_clear_all = QtWidgets.QPushButton(parent_widget)
            self.btn_clear_all.setGeometry(QtCore.QRect(1080, 5, 120, 27))
            self.btn_clear_all.setObjectName("btn_clear_all")
            self.btn_clear_all.setText("Clear All")
            self.btn_clear_all.setToolTip(
                "Reset every text box, radio button, check box and spin "
                "box back to its original default state.")
            # Connect only when WE create the button. If main_ui.py
            # already created it, it also wires the signal, so we avoid
            # a double connection here.
            self.btn_clear_all.clicked.connect(self.btn_clear_all_clicked)
        # Make sure the button sits on top of the header label.
        self.btn_clear_all.raise_()

        # --- Keyboard shortcut (Ctrl+Enter) for the Run button ---
        # We bind both 'Return' (main keyboard Enter) and 'Enter'
        # (numeric-keypad Enter) so the shortcut works from either key.
        # The QShortcut instances use window-wide context so they fire
        # even when a child widget (e.g. a QLineEdit) has focus.
        run_tooltip_suffix = " (shortcut: Ctrl+Enter)"
        existing_tooltip = self.btn_run.toolTip() or ""
        if run_tooltip_suffix.strip() not in existing_tooltip:
            self.btn_run.setToolTip(existing_tooltip + run_tooltip_suffix)
        # Native button shortcut (lets Qt expose it in the platform's way).
        self.btn_run.setShortcut(QKeySequence("Ctrl+Return"))
        # Extra window-level shortcuts for robustness across focus states
        # and for the numeric-keypad Enter key.
        self._run_shortcut_return = QShortcut(QKeySequence("Ctrl+Return"), self)
        self._run_shortcut_return.setContext(QtCore.Qt.WindowShortcut)
        self._run_shortcut_return.activated.connect(self.btn_run.click)
        self._run_shortcut_enter = QShortcut(QKeySequence("Ctrl+Enter"), self)
        self._run_shortcut_enter.setContext(QtCore.Qt.WindowShortcut)
        self._run_shortcut_enter.activated.connect(self.btn_run.click)

        self.center_screen()

    def btn_clear_all_clicked(self):
        """Reset every user-editable control on the dialog to its default
        value. This covers all QLineEdit text boxes, all QCheckBox metric
        selectors, both QRadioButton groups (ground-truth format and
        detection format) and the IOU threshold QDoubleSpinBox. The
        internally tracked directory / file path attributes are reset as
        well so the dialog is in the same state as right after launch."""
        # --- Clear all text boxes (QLineEdit) ---
        for line_edit in (self.txb_gt_dir,
                          self.txb_gt_images_dir,
                          self.txb_classes_gt,
                          self.txb_det_dir,
                          self.txb_classes_det,
                          self.txb_output_dir):
            line_edit.clear()

        # --- Reset radio buttons (ground-truth coordinates format) ---
        # Default: COCO (.json)
        self.rad_gt_format_coco_json.setChecked(True)
        for rb in (self.rad_gt_format_pascalvoc_xml,
                   self.rad_gt_format_labelme_xml,
                   self.rad_gt_format_openimages_csv,
                   self.rad_gt_format_imagenet_xml,
                   self.rad_gt_format_yolo_text,
                   self.rad_gt_format_abs_values_text,
                   self.rad_gt_format_cvat_xml):
            rb.setChecked(False)

        # --- Reset radio buttons (detection coordinates format) ---
        # Default: <class_id> <confidence> <x_center> <y_center> <width> <height> (RELATIVE)
        self.rad_det_ci_format_text_yolo_rel.setChecked(True)
        for rb in (self.rad_det_ci_format_text_xyx2y2_abs,
                   self.rad_det_ci_format_text_xywh_abs,
                   self.rad_det_format_coco_json,
                   self.rad_det_cn_format_text_yolo_rel,
                   self.rad_det_cn_format_text_xyx2y2_abs,
                   self.rad_det_cn_format_text_xywh_abs):
            rb.setChecked(False)

        # --- Reset all metric check boxes to their default (all checked) ---
        for cb in (self.chb_metric_AP_coco,
                   self.chb_metric_AP50_coco,
                   self.chb_metric_AP75_coco,
                   self.chb_metric_APsmall_coco,
                   self.chb_metric_APmedium_coco,
                   self.chb_metric_APlarge_coco,
                   self.chb_metric_AR_max1,
                   self.chb_metric_AR_max10,
                   self.chb_metric_AR_max100,
                   self.chb_metric_AR_small,
                   self.chb_metric_AR_medium,
                   self.chb_metric_AR_large,
                   self.chb_metric_AP_pascal,
                   self.chb_metric_mAP_pascal):
            cb.setChecked(True)

        # --- Reset the IOU threshold spin box ---
        self.dsb_IOU_pascal.setValue(0.5)

        # --- Reset the internally tracked paths so logic matches the UI ---
        self.dir_annotations_gt = None
        self.dir_images_gt = None
        self.filepath_classes_gt = None
        self.dir_dets = None
        self.filepath_classes_det = None
        self.dir_save_results = None

    def center_screen(self):
        size = self.size()
        desktopSize = QtWidgets.QDesktopWidget().screenGeometry()
        top = int((desktopSize.height() / 2) - (size.height() / 2))
        left = int((desktopSize.width() / 2) - (size.width() / 2))
        self.move(left, top)

    def closeEvent(self, event):
        conf = self.show_popup('Are you sure you want to close the program?',
                               'Closing',
                               buttons=QMessageBox.Yes | QMessageBox.No,
                               icon=QMessageBox.Question)
        if conf == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def show_popup(self,
                   message,
                   title,
                   buttons=QMessageBox.Ok | QMessageBox.Cancel,
                   icon=QMessageBox.Information):
        self.msgBox.setIcon(icon)
        self.msgBox.setText(message)
        self.msgBox.setWindowTitle(title)
        self.msgBox.setStandardButtons(buttons)
        return self.msgBox.exec()

    def load_annotations_gt(self):
        ret = []
        if self.rad_gt_format_coco_json.isChecked():
            ret = converter.coco2bb(self.dir_annotations_gt)
        elif self.rad_gt_format_cvat_xml.isChecked():
            ret = converter.cvat2bb(self.dir_annotations_gt)
        elif self.rad_gt_format_openimages_csv.isChecked():
            ret = converter.openimage2bb(self.dir_annotations_gt, self.dir_images_gt,
                                         BBType.GROUND_TRUTH)
        elif self.rad_gt_format_labelme_xml.isChecked():
            ret = converter.labelme2bb(self.dir_annotations_gt)
        elif self.rad_gt_format_pascalvoc_xml.isChecked():
            ret = converter.vocpascal2bb(self.dir_annotations_gt)
        elif self.rad_gt_format_imagenet_xml.isChecked():
            ret = converter.imagenet2bb(self.dir_annotations_gt)
        elif self.rad_gt_format_abs_values_text.isChecked():
            ret = converter.text2bb(self.dir_annotations_gt, bb_type=BBType.GROUND_TRUTH)
        elif self.rad_gt_format_yolo_text.isChecked():
            ret = converter.yolo2bb(self.dir_annotations_gt,
                                    self.dir_images_gt,
                                    self.filepath_classes_gt,
                                    bb_type=BBType.GROUND_TRUTH)
        # Make all types as GT
        [bb.set_bb_type(BBType.GROUND_TRUTH) for bb in ret]
        return ret

    def validate_det_choices(self):
        # If relative format was required, directory with images have to be valid
        if self.rad_det_ci_format_text_yolo_rel.isChecked(
        ) or self.rad_det_cn_format_text_yolo_rel.isChecked():
            # Verify if directory with images was provided
            valid_image_dir = False
            if self.dir_images_gt is not None and os.path.isdir(self.dir_images_gt):
                found_image_files = general_utils.get_files_dir(
                    self.dir_images_gt, extensions=['jpg', 'jpge', 'png', 'bmp', 'tiff', 'tif'])
                if len(found_image_files) != 0:
                    valid_image_dir = True
            if valid_image_dir is False:
                self.show_popup(
                    f'For the selected annotation type, it is necessary to inform a directory with the dataset images.\nDirectory is empty or does not have valid images.',
                    'Invalid image directory',
                    buttons=QMessageBox.Ok,
                    icon=QMessageBox.Information)
                return False
        # if its detection format requires class_id, text file containing the classes of objects must be informed
        if self.rad_det_ci_format_text_yolo_rel.isChecked(
        ) or self.rad_det_ci_format_text_xyx2y2_abs.isChecked(
        ) or self.rad_det_ci_format_text_xywh_abs.isChecked():
            # Verify if text file with classes was provided
            valid_txt_file = False
            if self.filepath_classes_det is not None and os.path.isfile(self.filepath_classes_det):
                classes = general_utils.get_classes_from_txt_file(self.filepath_classes_det)
                if len(classes) != 0:
                    valid_txt_file = True
            if valid_txt_file is False:
                self.show_popup(
                    f'For the selected annotation type, it is necessary to inform a valid text file listing one class per line.\nCheck if the path for the .txt file is correct and if it contains at least one class.',
                    'Invalid text file or not found',
                    buttons=QMessageBox.Ok,
                    icon=QMessageBox.Information)
                return False
        return True

    def load_annotations_det(self):
        ret = []
        if not self.validate_det_choices():
            return ret, False

        if self.rad_det_format_coco_json.isChecked():
            ret = converter.coco2bb(self.dir_dets, bb_type=BBType.DETECTED)
        elif self.rad_det_ci_format_text_yolo_rel.isChecked(
        ) or self.rad_det_cn_format_text_yolo_rel.isChecked():
            ret = converter.text2bb(self.dir_dets,
                                    bb_type=BBType.DETECTED,
                                    bb_format=BBFormat.YOLO,
                                    type_coordinates=CoordinatesType.RELATIVE,
                                    img_dir=self.dir_images_gt)
        elif self.rad_det_ci_format_text_xyx2y2_abs.isChecked(
        ) or self.rad_det_cn_format_text_xyx2y2_abs.isChecked():
            ret = converter.text2bb(self.dir_dets,
                                    bb_type=BBType.DETECTED,
                                    bb_format=BBFormat.XYX2Y2,
                                    type_coordinates=CoordinatesType.ABSOLUTE,
                                    img_dir=self.dir_images_gt)
        elif self.rad_det_ci_format_text_xywh_abs.isChecked(
        ) or self.rad_det_cn_format_text_xywh_abs.isChecked():
            ret = converter.text2bb(self.dir_dets,
                                    bb_type=BBType.DETECTED,
                                    bb_format=BBFormat.XYWH,
                                    type_coordinates=CoordinatesType.ABSOLUTE,
                                    img_dir=self.dir_images_gt)
        # Verify if for the selected format, detections were found
        if len(ret) == 0:
            self.show_popup(
                'No file was found for the selected detection format in the annotations directory.',
                'No file was found',
                buttons=QMessageBox.Ok,
                icon=QMessageBox.Information)
            return ret, False

        # If detection requires class_id, replace the detection names (integers) by a class from the txt file
        if self.rad_det_ci_format_text_yolo_rel.isChecked(
        ) or self.rad_det_ci_format_text_xyx2y2_abs.isChecked(
        ) or self.rad_det_ci_format_text_xywh_abs.isChecked():
            ret = general_utils.replace_id_with_classes(ret, self.filepath_classes_det)
        return ret, True

    def btn_gt_statistics_clicked(self):
        # If yolo format is selected, file with classes must be informed
        if self.rad_gt_format_yolo_text.isChecked():
            if self.filepath_classes_gt is None or os.path.isfile(
                    self.filepath_classes_gt) is False:
                self.show_popup(
                    'For the selected groundtruth format, a valid file with classes must be informed.',
                    'Invalid file',
                    buttons=QMessageBox.Ok,
                    icon=QMessageBox.Information)
                return

        gt_annotations = self.load_annotations_gt()
        if gt_annotations is None or len(gt_annotations) == 0:
            self.show_popup(
                'Directory with ground-truth annotations was not specified or do not contain annotations in the chosen format.',
                'Annotations not found',
                buttons=QMessageBox.Ok,
                icon=QMessageBox.Information)
            return
        if self.dir_images_gt is None:
            self.show_popup(
                'Directory with ground-truth images was not specified or do not contain images.',
                'Images not found',
                buttons=QMessageBox.Ok,
                icon=QMessageBox.Information)
            return
        # Open statistics dialot on the gt annotations
        self.dialog_statistics.show_dialog(BBType.GROUND_TRUTH, gt_annotations, None,
                                           self.dir_images_gt)

    def btn_gt_dir_clicked(self):
        if self.txb_gt_dir.text() == '':
            txt = self.current_directory
        else:
            txt = self.txb_gt_dir.text()
        directory = QFileDialog.getExistingDirectory(
            self, 'Choose directory with ground truth annotations', txt)
        if directory == '':
            return
        if os.path.isdir(directory):
            self.txb_gt_dir.setText(directory)
            self.dir_annotations_gt = directory
        else:
            self.dir_annotations_gt = None

    def btn_gt_classes_clicked(self):
        filepath = QFileDialog.getOpenFileName(self, 'Choose a file with a list of classes',
                                               self.current_directory,
                                               "Image files (*.txt *.names)")
        filepath = filepath[0]
        if os.path.isfile(filepath):
            self.txb_classes_gt.setText(filepath)
            self.filepath_classes_gt = filepath
        else:
            self.filepath_classes_gt = None

    def btn_gt_images_dir_clicked(self):
        if self.txb_gt_images_dir.text() == '':
            txt = self.current_directory
        else:
            txt = self.txb_gt_images_dir.text()
        directory = QFileDialog.getExistingDirectory(self,
                                                     'Choose directory with ground truth images',
                                                     txt)
        if directory != '':
            self.txb_gt_images_dir.setText(directory)
            self.dir_images_gt = directory

    def btn_det_classes_clicked(self):
        filepath = QFileDialog.getOpenFileName(self, 'Choose a file with a list of classes',
                                               self.current_directory,
                                               "Image files (*.txt *.names)")
        filepath = filepath[0]
        if os.path.isfile(filepath):
            self.txb_classes_det.setText(filepath)
            self.filepath_classes_det = filepath
        else:
            self.filepath_classes_det = None
            self.txb_classes_det.setText('')

    def btn_det_dir_clicked(self):
        if self.txb_det_dir.text() == '':
            txt = self.current_directory
        else:
            txt = self.txb_det_dir.text()
        directory = QFileDialog.getExistingDirectory(self, 'Choose directory with detections', txt)
        if directory == '':
            return
        if os.path.isdir(directory):
            self.txb_det_dir.setText(directory)
            self.dir_dets = directory
        else:
            self.dir_dets = None

    def btn_statistics_det_clicked(self):
        det_annotations, passed = self.load_annotations_det()
        if passed is False:
            return
        gt_annotations = self.load_annotations_gt()
        if self.dir_images_gt is None or os.path.isdir(self.dir_images_gt) is False:
            self.show_popup(
                'Directory with ground-truth images was not specified or do not contain images.',
                'Images not found',
                buttons=QMessageBox.Ok,
                icon=QMessageBox.Information)
            return
        # Open statistics dialog on the detections
        self.dialog_statistics.show_dialog(BBType.DETECTED, gt_annotations, det_annotations,
                                           self.dir_images_gt)

    def btn_output_dir_clicked(self):
        if self.txb_output_dir.text() == '':
            txt = self.current_directory
        else:
            txt = self.txb_output_dir.text()
        directory = QFileDialog.getExistingDirectory(self, 'Choose directory to save the results',
                                                     txt)
        if os.path.isdir(directory):
            self.txb_output_dir.setText(directory)
            self.dir_save_results = directory
        else:
            self.dir_save_results = None

    def btn_run_clicked(self):
        if self.dir_save_results is None or os.path.isdir(self.dir_save_results) is False:
            self.show_popup('Output directory to save results was not specified or does not exist.',
                            'Invalid output directory',
                            buttons=QMessageBox.Ok,
                            icon=QMessageBox.Information)
            return
        # Get detections
        det_annotations, passed = self.load_annotations_det()
        if passed is False:
            return
        # Verify if there are detections
        if det_annotations is None or len(det_annotations) == 0:
            self.show_popup(
                'No detection of the selected type was found in the folder.\nCheck if the selected type corresponds to the files in the folder and try again.',
                'Invalid detections',
                buttons=QMessageBox.Ok,
                icon=QMessageBox.Information)
            return

        gt_annotations = self.load_annotations_gt()
        if gt_annotations is None or len(gt_annotations) == 0:
            self.show_popup(
                'No ground-truth bounding box of the selected type was found in the folder.\nCheck if the selected type corresponds to the files in the folder and try again.',
                'Invalid groundtruths',
                buttons=QMessageBox.Ok,
                icon=QMessageBox.Information)
            return

        coco_res = {}
        pascal_res = {}
        # If any coco metric is required
        if self.chb_metric_AP_coco.isChecked() or self.chb_metric_AP50_coco.isChecked(
        ) or self.chb_metric_AP75_coco.isChecked() or self.chb_metric_APsmall_coco.isChecked(
        ) or self.chb_metric_APmedium_coco.isChecked() or self.chb_metric_APlarge_coco.isChecked(
        ) or self.chb_metric_AR_max1.isChecked() or self.chb_metric_AR_max10.isChecked(
        ) or self.chb_metric_AR_max100.isChecked() or self.chb_metric_AR_small.isChecked(
        ) or self.chb_metric_AR_medium.isChecked() or self.chb_metric_AR_large.isChecked():
            coco_res = get_coco_summary(gt_annotations, det_annotations)
            # Remove not checked metrics
            if not self.chb_metric_AP_coco.isChecked():
                del coco_res['AP']
            if not self.chb_metric_AP50_coco.isChecked():
                del coco_res['AP50']
            if not self.chb_metric_AP75_coco.isChecked():
                del coco_res['AP75']
            if not self.chb_metric_APsmall_coco.isChecked():
                del coco_res['APsmall']
            if not self.chb_metric_APmedium_coco.isChecked():
                del coco_res['APmedium']
            if not self.chb_metric_APlarge_coco.isChecked():
                del coco_res['APlarge']
            if not self.chb_metric_AR_max1.isChecked():
                del coco_res['AR1']
            if not self.chb_metric_AR_max10.isChecked():
                del coco_res['AR10']
            if not self.chb_metric_AR_max100.isChecked():
                del coco_res['AR100']
            if not self.chb_metric_AR_small.isChecked():
                del coco_res['ARsmall']
            if not self.chb_metric_AR_medium.isChecked():
                del coco_res['ARmedium']
            if not self.chb_metric_AR_large.isChecked():
                del coco_res['ARlarge']
        # If any pascal metric is required
        if self.chb_metric_AP_pascal.isChecked() or self.chb_metric_mAP_pascal.isChecked():
            iou_threshold = self.dsb_IOU_pascal.value()
            pascal_res = get_pascalvoc_metrics(gt_annotations,
                                               det_annotations,
                                               iou_threshold=iou_threshold,
                                               generate_table=True)
            mAP = pascal_res['mAP']

            if not self.chb_metric_AP_pascal.isChecked():
                del pascal_res['per_class']
            if not self.chb_metric_mAP_pascal.isChecked():
                del pascal_res['mAP']

            if 'per_class' in pascal_res:
                # Save a single plot with all classes
                plot_precision_recall_curve(pascal_res['per_class'],
                                            mAP=mAP,
                                            savePath=self.dir_save_results,
                                            showGraphic=False)
                # Save plots for each class
                plot_precision_recall_curves(pascal_res['per_class'],
                                             showAP=True,
                                             savePath=self.dir_save_results,
                                             showGraphic=False)

        if len(coco_res) + len(pascal_res) == 0:
            self.show_popup('No results to show',
                            'No results',
                            buttons=QMessageBox.Ok,
                            icon=QMessageBox.Information)
        else:
            self.dialog_results.show_dialog(coco_res, pascal_res, self.dir_save_results)
