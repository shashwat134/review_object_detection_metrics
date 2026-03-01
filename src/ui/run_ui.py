"""
run_ui.py
=========

Main application window for the *Object Detection Metrics* GUI.

This module wires the auto-generated Qt Designer layout
(:mod:`src.ui.main_ui`) together with the evaluation back-end
(:mod:`src.evaluators`) and the annotation converters
(:mod:`src.utils.converter`).

User-facing additions implemented in this file
-----------------------------------------------

* **Clear All** button
      A single click restores every input control (text fields,
      radio buttons, check boxes and the IOU spin box) to the same
      state the dialog has immediately after launch.

* **Blue background**
      A light-blue window background (``#cfe8ff``) applied through a
      Qt stylesheet so the colour renders consistently across
      desktop themes and platforms.

* **Keyboard shortcuts**
      ===============  ==========================================
      Shortcut         Action
      ===============  ==========================================
      ``Ctrl+Enter``   Trigger the **RUN** button
      ``Ctrl+Q``       Quit the application (with confirmation)
      ===============  ==========================================

All customisations live in :class:`Main_Dialog` so the generated
``main_ui.py`` can be regenerated from ``main_ui.ui`` without
losing behaviour.
"""

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


# ---------------------------------------------------------------------------
# Visual / interaction constants
# ---------------------------------------------------------------------------

#: Light-blue colour used for the main-window background.  Chosen to be
#: bright enough that the existing black label text keeps a comfortable
#: contrast ratio, while still being obviously distinct from the default
#: system grey.
WINDOW_BACKGROUND_COLOR = "#cfe8ff"

#: Geometry of the **Clear All** button expressed as (x, y, width, height)
#: in pixels within the fixed-size 1214x850 main dialog.  The position
#: ``(1080, 5)`` places the button in the top-right corner, aligned with
#: the "Ground truth" section header, without overlapping any existing
#: control.
CLEAR_ALL_BUTTON_GEOMETRY = (1080, 5, 120, 27)

#: Default value for the Pascal VOC IOU-threshold spin box.  Mirrors the
#: value baked into the Designer (.ui) file so :meth:`Main_Dialog.
#: btn_clear_all_clicked` can restore it symbolically rather than with a
#: magic number.
DEFAULT_IOU_THRESHOLD = 0.5


class Main_Dialog(QMainWindow, Main_UI):
    """
    Top-level window of the *Object Detection Metrics* application.

    The class multiply-inherits from :class:`~PyQt5.QtWidgets.QMainWindow`
    (to obtain a proper top-level window) and from the auto-generated
    ``Ui_Dialog`` mix-in (aliased as ``Main_UI``) that lays out every
    widget defined in ``main_ui.ui``.

    Responsibilities
    ----------------
    * Construct and theme the window (blue background, **Clear All**
      button, keyboard shortcuts).
    * Convert user input into calls against the evaluator back-end.
    * Surface results and validation feedback through secondary dialogs.

    Keyboard shortcuts
    ------------------
    =============  ====================================================
    ``Ctrl+Enter``  Click the **RUN** button.
    ``Ctrl+Q``      Request application shutdown.  Routed through
                    :meth:`closeEvent`, so the user still sees the
                    "Are you sure?" confirmation prompt.
    =============  ====================================================

    Notes
    -----
    Qt treats the main-keyboard *Enter* key (``Qt.Key_Return``) and the
    numeric-keypad *Enter* key (``Qt.Key_Enter``) as distinct keys.  Both
    variants are registered for the Run shortcut so users do not have to
    care which physical key they press.
    """

    def __init__(self):
        """
        Build the main window.

        After the auto-generated :meth:`setupUi` has placed every widget
        defined in Qt Designer, the constructor applies the runtime
        customisations in a fixed, easy-to-audit order:

        1. :meth:`_apply_background_style`    - paint the window blue.
        2. :meth:`_install_clear_all_button`  - add / wire the reset button.
        3. :meth:`_install_run_shortcut`      - bind ``Ctrl+Enter`` to RUN.
        4. :meth:`_install_quit_shortcut`     - bind ``Ctrl+Q`` to quit.

        Splitting the work into helpers keeps each concern self-contained
        and individually testable.
        """
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

        # ---- Runtime UI customisations (order matters only for z-stacking) ----
        self._apply_background_style()
        self._install_clear_all_button()
        self._install_run_shortcut()
        self._install_quit_shortcut()

        self.center_screen()

    # ------------------------------------------------------------------ #
    #  Private setup helpers                                             #
    # ------------------------------------------------------------------ #

    def _apply_background_style(self):
        """
        Paint the entire window background in :data:`WINDOW_BACKGROUND_COLOR`.

        Implementation detail
        ~~~~~~~~~~~~~~~~~~~~~
        When ``setupUi`` is called on a :class:`QMainWindow`, the Designer
        widgets end up parented to an implicitly created *central widget*,
        not to the main window itself.  Applying a stylesheet only to the
        ``QMainWindow`` can therefore be masked by that central widget on
        some Qt styles.  To guarantee a consistent look we:

        1. Set a selector-scoped stylesheet on the main window so only the
           window surface is affected (child widgets such as line edits
           keep their own backgrounds).
        2. Additionally style the central widget directly.
        3. Fall back to a :class:`QPalette` paint in the unlikely event
           that no central widget exists.
        """
        css = "background-color: %s;" % WINDOW_BACKGROUND_COLOR

        # Selector limits the rule to the QMainWindow surface; without it
        # every descendant (QLineEdit, QPushButton, ...) would inherit the
        # blue background, which is visually noisy.
        self.setStyleSheet("QMainWindow { %s }" % css)

        central = self.centralWidget()
        if central is not None:
            central.setStyleSheet(css)
        else:
            # Defensive branch: setupUi normally provides a central
            # widget, but if a future refactor removes it we still want a
            # blue window rather than silently reverting to system grey.
            palette = self.palette()
            palette.setColor(QtGui.QPalette.Window,
                             QtGui.QColor(WINDOW_BACKGROUND_COLOR))
            self.setPalette(palette)
            self.setAutoFillBackground(True)

    def _install_clear_all_button(self):
        """
        Ensure a **Clear All** push button exists and is wired up.

        Two creation paths are supported:

        * **Designer path** - ``main_ui.py`` (regenerated from
          ``main_ui.ui``) already creates ``btn_clear_all`` and connects
          its ``clicked`` signal to :meth:`btn_clear_all_clicked`.  In
          this case we only raise the button to the top of the z-order
          so it is not hidden behind the wide header label.

        * **Fallback path** - an older or stale ``main_ui.py`` may not
          yet contain the button.  We create and connect it here so the
          feature is still available.  Connecting only on this path
          prevents a *double* signal-slot connection (which would make
          the reset handler run twice per click).
        """
        if not hasattr(self, "btn_clear_all") or self.btn_clear_all is None:
            # -- Fallback path: create the widget ourselves --------------
            parent = self.centralWidget() if self.centralWidget() is not None else self
            self.btn_clear_all = QtWidgets.QPushButton(parent)
            self.btn_clear_all.setGeometry(
                QtCore.QRect(*CLEAR_ALL_BUTTON_GEOMETRY))
            self.btn_clear_all.setObjectName("btn_clear_all")
            self.btn_clear_all.setText("Clear All")
            self.btn_clear_all.setToolTip(
                "Reset every text box, radio button, check box and the "
                "IOU spin box to the values they had on application "
                "start-up.")
            # Wire the click handler.  Done only on the fallback path to
            # avoid double-connecting when the Designer code already did.
            self.btn_clear_all.clicked.connect(self.btn_clear_all_clicked)

        # The "Ground truth" header label spans the full window width at
        # y=10.  Raise the button so it is drawn *above* the label rather
        # than clipped behind it.
        self.btn_clear_all.raise_()

    def _install_run_shortcut(self):
        """
        Bind ``Ctrl+Enter`` to the **RUN** button.

        Why three bindings?
        ~~~~~~~~~~~~~~~~~~~
        * :meth:`QPushButton.setShortcut` gives Qt a chance to surface the
          accelerator through native means (e.g. underline hints,
          accessibility APIs).  This only responds to the *Return* key.

        * A button-level shortcut is swallowed when keyboard focus sits
          inside certain child widgets.  Two additional :class:`QShortcut`
          objects with :data:`~QtCore.Qt.WindowShortcut` scope therefore
          provide a reliable, focus-independent trigger.

        * The two window-level shortcuts cover *both* physical Enter keys:
          ``Ctrl+Return`` for the main-keyboard key and ``Ctrl+Enter`` for
          the numeric-keypad key.  Qt treats them as distinct key codes,
          so two separate bindings are required.

        All routes converge on :meth:`QPushButton.click`, which emits the
        button's ``clicked`` signal exactly as a mouse press would -
        including the visual pressed-state animation.
        """
        # Append a visible hint to the existing tooltip (only once - guard
        # against repeated calls during unit tests).
        hint = " (shortcut: Ctrl+Enter)"
        tooltip = self.btn_run.toolTip() or ""
        if hint.strip() not in tooltip:
            self.btn_run.setToolTip(tooltip + hint)

        # (1) Native button accelerator.
        self.btn_run.setShortcut(QKeySequence("Ctrl+Return"))

        # (2) Window-scope shortcut for the main-keyboard Enter key.
        self._run_shortcut_return = QShortcut(
            QKeySequence("Ctrl+Return"), self)
        self._run_shortcut_return.setContext(QtCore.Qt.WindowShortcut)
        self._run_shortcut_return.activated.connect(self.btn_run.click)

        # (3) Window-scope shortcut for the numeric-keypad Enter key.
        self._run_shortcut_enter = QShortcut(
            QKeySequence("Ctrl+Enter"), self)
        self._run_shortcut_enter.setContext(QtCore.Qt.WindowShortcut)
        self._run_shortcut_enter.activated.connect(self.btn_run.click)

    def _install_quit_shortcut(self):
        """
        Bind ``Ctrl+Q`` to an orderly application shutdown.

        The shortcut calls :meth:`QWidget.close`, which in turn delivers a
        :class:`QCloseEvent` to :meth:`closeEvent`.  That override already
        prompts the user with an *"Are you sure?"* confirmation dialog and
        honours their Yes / No answer, so ``Ctrl+Q`` enjoys the exact same
        safety net as clicking the window-manager close button.

        :data:`QKeySequence.Quit` is used instead of a hard-coded string so
        macOS users automatically get the idiomatic ``Cmd+Q`` mapping;
        Linux and Windows resolve it to ``Ctrl+Q``.
        """
        self._quit_shortcut = QShortcut(
            QKeySequence(QKeySequence.Quit), self)
        self._quit_shortcut.setContext(QtCore.Qt.WindowShortcut)
        self._quit_shortcut.activated.connect(self.close)

        # Some Qt builds on Windows / Linux leave `QKeySequence.Quit` empty
        # (no platform default).  Provide an explicit Ctrl+Q fallback so
        # the advertised shortcut is always honoured.
        if self._quit_shortcut.key().isEmpty():
            self._quit_shortcut.setKey(QKeySequence("Ctrl+Q"))
        else:
            # Platform supplied a mapping (e.g. Cmd+Q on macOS).  Still
            # register a literal Ctrl+Q so documentation remains accurate
            # on every OS.
            self._quit_shortcut_ctrlq = QShortcut(
                QKeySequence("Ctrl+Q"), self)
            self._quit_shortcut_ctrlq.setContext(QtCore.Qt.WindowShortcut)
            self._quit_shortcut_ctrlq.activated.connect(self.close)

    # ------------------------------------------------------------------ #
    #  Public slots                                                      #
    # ------------------------------------------------------------------ #

    def btn_clear_all_clicked(self):
        """
        Restore the dialog to its pristine, just-launched state.

        Every user-editable widget is reverted to the default value that
        ``main_ui.ui`` defines, and every internally cached path attribute
        is cleared so validation logic behaves as if the user had typed
        nothing yet.

        Widgets reset by this slot
        --------------------------
        ========================  ===================================
        Widget type               Widgets affected
        ========================  ===================================
        ``QLineEdit``             ``txb_gt_dir``, ``txb_gt_images_dir``,
                                  ``txb_classes_gt``, ``txb_det_dir``,
                                  ``txb_classes_det``, ``txb_output_dir``
        ``QRadioButton`` (GT)     Default -> *COCO (.json)*
        ``QRadioButton`` (Det)    Default -> *<class_id> ... (RELATIVE)*
        ``QCheckBox``             All fourteen metric toggles -> checked
        ``QDoubleSpinBox``        ``dsb_IOU_pascal`` -> ``0.5``
        ========================  ===================================

        Attributes reset to ``None``
        ----------------------------
        ``dir_annotations_gt``, ``dir_images_gt``, ``filepath_classes_gt``,
        ``dir_dets``, ``filepath_classes_det``, ``dir_save_results``

        Idempotency
        -----------
        The slot is safe to invoke repeatedly; the second and subsequent
        calls are no-ops because every control is already at its default.
        """
        # 1. Text boxes --------------------------------------------------
        #    These hold directory / file paths selected via file dialogs.
        #    ``.clear()`` both empties the visible text and (because the
        #    widgets are read-only) cannot be undone by the user through
        #    the keyboard, so no partially-cleared state can linger.
        for line_edit in (self.txb_gt_dir,
                          self.txb_gt_images_dir,
                          self.txb_classes_gt,
                          self.txb_det_dir,
                          self.txb_classes_det,
                          self.txb_output_dir):
            line_edit.clear()

        # 2. Ground-truth format radio group -----------------------------
        #    Auto-exclusivity means selecting the default automatically
        #    deselects whatever was previously checked, but we still loop
        #    over the remaining buttons for explicitness and to guard
        #    against future changes to the exclusivity setting.
        self.rad_gt_format_coco_json.setChecked(True)
        for rb in (self.rad_gt_format_pascalvoc_xml,
                   self.rad_gt_format_labelme_xml,
                   self.rad_gt_format_openimages_csv,
                   self.rad_gt_format_imagenet_xml,
                   self.rad_gt_format_yolo_text,
                   self.rad_gt_format_abs_values_text,
                   self.rad_gt_format_cvat_xml):
            rb.setChecked(False)

        # 3. Detection format radio group --------------------------------
        self.rad_det_ci_format_text_yolo_rel.setChecked(True)
        for rb in (self.rad_det_ci_format_text_xyx2y2_abs,
                   self.rad_det_ci_format_text_xywh_abs,
                   self.rad_det_format_coco_json,
                   self.rad_det_cn_format_text_yolo_rel,
                   self.rad_det_cn_format_text_xyx2y2_abs,
                   self.rad_det_cn_format_text_xywh_abs):
            rb.setChecked(False)

        # 4. Metric check boxes ------------------------------------------
        #    The Designer default is "all metrics enabled"; matching that
        #    here keeps Clear All intuitive (a reset, not a blank slate).
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

        # 5. IOU threshold spin box --------------------------------------
        self.dsb_IOU_pascal.setValue(DEFAULT_IOU_THRESHOLD)

        # 6. Internal cache ----------------------------------------------
        #    These mirrors are what the evaluator methods actually read.
        #    Leaving them populated while the UI shows empty fields would
        #    let a user accidentally re-run against stale paths.
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
